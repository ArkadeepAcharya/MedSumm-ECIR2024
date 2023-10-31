import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM, AutoImageProcessor, ViTModel
import os, sys
from pytorchtools import EarlyStopping
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import numpy as np





device = torch.device("cpu") 

df_train = pd.read_csv("train.csv")
df_val = pd.read_csv("val.csv")
df_test = pd.read_csv("test.csv")

model_name = "path to llama-2 weights"



tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")


img_path = ""




class MMCQSD(torch.utils.data.Dataset):
    """Uses jsonl data to preprocess and serve 
    dictionary of multimodal tensors for model input.
    """

    def __init__(
        self,
      	data,
      	img_path,
    ):

        self.data = data
        self.img_path = img_path

        self.transforms =  T.Compose([
                                             T.Resize(256),
                                             T.CenterCrop(224),
                                             T.ToTensor(),
                                          ])


       
        
    def __len__(self):
        """This method is called when you do len(instance) 
        for an instance of this class.
        """
        return len(self.data)

    def __getitem__(self, idx):
   
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.data["new_relative_path"][idx]

        input_image= self.transforms(Image.open(os.path.join(self.img_path,img_id)))
        
        input_text = str(self.data["Hinglish_Question"][idx]) +"SUMMARY: " 
        #print(input_image.shape)

        if input_image.shape[0]<3:       
          input_image = torch.cat([input_image,input_image,input_image],dim=0)
        else:
          input_image = input_image[:3,:,:]

        print(input_image.shape)
        
        sample = {
        "img_id":img_id,
        "image":input_image,
        "text": input_text,
        }


        return sample


bm_dataset_test = MMCQSD(df_test,img_path)
dataloader_test = DataLoader(bm_dataset_test, batch_size=1,shuffle=False, num_workers=0)
print("test data loaded")





class MultimodalLLM(nn.Module):
  def __init__(self,model_name):
    super(MultimodalLLM,self).__init__()
    self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    for name, param in self.vit_model.named_parameters():
      param.requires_grad = False

    self.projection = nn.Linear(768,4096)

    self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    self.tokenizer.pad_token = self.tokenizer.eos_token



    self.quant_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16
                        )

    self.llm_model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=self.quant_config)

    for name, param in self.llm_model.named_parameters():
      param.requires_grad = False


    self.llm_model.config.use_cache = False

    self.kbit_model = prepare_model_for_kbit_training(self.llm_model)


    self.config = LoraConfig(
            r=8, 
            lora_alpha=32, 
            #target_modules=["query_key_value"],
            #target_modules = ["q_proj", "v_proj"],
            lora_dropout=0.05, 
            bias="none", 
            task_type="CAUSAL_LM"
        )

    self.adapter_model = get_peft_model(self.kbit_model, self.config)

    self.max_length = 512


  def visual_encoder(self,image):
    inputs = self.image_processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = self.vit_model(pixel_values = inputs.pixel_values.to(device))

    last_hidden_states = outputs.last_hidden_state
    # print(last_hidden_states.shape)

    img_proj = self.projection(last_hidden_states)
    

    att_img = torch.ones([img_proj.shape[0], img_proj.shape[1]+1],
                         dtype=inputs.pixel_values.dtype,
                         device=device) 


    return att_img, img_proj


  def forward(self,text,image):



    self.tokenizer.padding_side = "right"

    att_img, img_proj = self.visual_encoder(image)

    tokenized_text = self.tokenizer(text, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_length,  add_special_tokens=False)

    text_embeds = self.adapter_model.model.model.embed_tokens(tokenized_text.input_ids.to(device))

    print(text_embeds.shape)


    bos = torch.ones([tokenized_text.input_ids.shape[0], 1],
                         dtype=tokenized_text.input_ids.dtype,
                         device=device) * self.tokenizer.bos_token_id


    bos_embeds = self.adapter_model.model.model.embed_tokens(bos)

    input_embeds = torch.cat([bos_embeds, img_proj, text_embeds.to(device)], dim=1)

    attention_mask = torch.cat([att_img.to(device), tokenized_text.attention_mask.to(device)],dim=1)



    targets = tokenized_text.input_ids.masked_fill(
                tokenized_text.input_ids == self.tokenizer.pad_token_id, -100
            )


    empty_targets = (
    torch.ones([img_proj.shape[0], img_proj.shape[1]+1], dtype=torch.long).to(device).fill_(-100)  # plus one for bos
        )

    targets = torch.cat([empty_targets.to(device), targets.to(device)], dim=1)



    outputs = self.adapter_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,)


    return outputs






class PredictLLM(nn.Module):
  def __init__(self,model_name):
    super(PredictLLM,self).__init__()

    self.gen_model = MultimodalLLM(model_name)

    self.gen_model.to(device)





  def forward(self,text,image,chk_file):

    self.gen_model.load_state_dict(torch.load(chk_file))

    for name, param in self.gen_model.named_parameters():
        param.requires_grad = False


    self.gen_model.tokenizer.padding_side = "right"

    att_img, img_proj = self.gen_model.visual_encoder(image)

    tokenized_text = self.gen_model.tokenizer(text, return_tensors="pt", padding="longest", truncation=True, max_length=self.gen_model.max_length,  add_special_tokens=False)

    text_embeds = self.gen_model.adapter_model.model.model.embed_tokens(tokenized_text.input_ids.to(device))

    #print(text_embeds.shape)


    bos = torch.ones([tokenized_text.input_ids.shape[0], 1],
                         dtype=tokenized_text.input_ids.dtype,
                         device=device) * self.gen_model.tokenizer.bos_token_id


    bos_embeds = self.gen_model.adapter_model.model.model.embed_tokens(bos)

    input_embeds = torch.cat([bos_embeds, img_proj, text_embeds.to(device)], dim=1)



    return generate_beam( gen_model = self.gen_model,
                tokenizer = self.gen_model.tokenizer,
                embed = input_embeds,
                )[0]





exp_name = "llama2"
exp_path = ""
chk_file =""
#  os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')


model_name = "path "
predictllm = PredictLLM(model_name)


 



def generate_beam(
    gen_model,
    tokenizer,
    beam_size: int = 5,
    embed=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):

    gen_model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(gen_model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():

        generated = embed

        for i in range(entry_length):
            outputs = gen_model.llm_model(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = gen_model.adapter_model.model.model.embed_tokens(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts
  
  

exp_name = "llama2"
exp_path = "checkpoint folder"


model_name = "path to llama 2 weights"
model = MultimodalLLM(model_name)

model.to(device)

print(next(model.parameters()).is_cuda)

lr = 0.00001


optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

import os, sys
#sys.path.append('path_to_the_module/early-stopping-pytorch')
from pytorchtools import EarlyStopping
#from torchsample.callbacks import EarlyStopping

chk_file = ""


model.load_state_dict(torch.load(chk_file))

i=0

for data in dataloader_test:
    i=i+1

    video_id = []
    video_text = []
    gen_response = []
    target=[]

    with torch.no_grad():
        id = data["img_id"]
        text = data["text"]
        image = data["image"]

        video_text.append(text)
        video_id.append(id)
        print(image.shape)
        # img_proj = model.projection(image)

        att_img, img_proj = model.visual_encoder(image)

        tokenized_text = model.tokenizer(text, return_tensors="pt", padding="longest", truncation=True, max_length=model.max_length,  add_special_tokens=False)

        text_embeds = model.adapter_model.model.model.embed_tokens(tokenized_text.input_ids.to(device))

        #print("text shape",text_embeds.shape)
        #print("image shape",img_proj.shape)


        bos = torch.ones([tokenized_text.input_ids.shape[0], 1],
                             dtype=tokenized_text.input_ids.dtype,
                             device=device) * model.tokenizer.bos_token_id


        bos_embeds = model.adapter_model.model.model.embed_tokens(bos)

        input_embeds = torch.cat([bos_embeds, img_proj, text_embeds.to(device)], dim=1)

        print("yes")


        outputs = model.llm_model.generate(inputs_embeds = input_embeds,max_length=128)

        response = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(response)
        gen_response.append(response)

        data = {}
 
        data["img_id"] = video_id
        data["Codemixed_Question"] = video_text
        data["response"] = gen_response
        
        df_generated = pd.DataFrame(data)
            
        df_generated.to_csv("generated_data_llama2_multimodal.csv",mode='a', index=False, header=False)
        print("Generated and saved for ",i)  




  















