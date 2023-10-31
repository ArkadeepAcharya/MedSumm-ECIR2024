import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM, AutoImageProcessor, ViTModel
import os, sys
from pytorchtools import EarlyStopping
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model,TaskType
from pathlib import Path
from PIL import Image
import torchvision.transforms as T


device = torch.device("cuda:0") 

df_train = pd.read_csv("train.csv")
df_val = pd.read_csv("val.csv")
df_test = pd.read_csv("test.csv")

model_name = "google/flan-t5-large"



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
        input_text ="SUMMARIZE THE FOLLOWING TEXT: "+ str(self.data["Hinglish_Question"][idx])
        target= " SUMMARY: "+ str(self.data['multimodal_Question_summ'][idx])

        

        if input_image.shape[0]<3:       
          input_image = torch.cat([input_image,input_image,input_image],dim=0)
        else:
          input_image = input_image[:3,:,:]
        
        sample = {
        "img_id":img_id,
        "image":input_image,
        "text": input_text,
        "target":target
        }


        return sample





bm_dataset_train = MMCQSD(df_train,img_path)
dataloader_train = DataLoader(bm_dataset_train, batch_size=1,shuffle=False, num_workers=0)

print("train_data loaded")

bm_dataset_val = MMCQSD(df_val,img_path)
dataloader_val = DataLoader(bm_dataset_val, batch_size=1,shuffle=False, num_workers=0)
print("validation_data loaded")


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

    self.projection = nn.Linear(768,1024)

    self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,cache_dir='')

    self.tokenizer.pad_token = self.tokenizer.eos_token


    self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name,load_in_8bit=True,device_map=device,cache_dir='')

    for name, param in self.llm_model.named_parameters():
      param.requires_grad = False


    self.llm_model.config.use_cache = False

    self.kbit_model = prepare_model_for_int8_training(self.llm_model)


    self.config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["q", "v"],
            lora_dropout=0.05, 
            bias="none", 
            task_type=TaskType.SEQ_2_SEQ_LM
        )

    self.adapter_model = get_peft_model(self.kbit_model, self.config)

    self.max_length = 512
    # 64


  def visual_encoder(self,image):
    inputs = self.image_processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = self.vit_model(pixel_values = inputs.pixel_values.to(device))

    last_hidden_states = outputs.last_hidden_state

    img_proj = self.projection(last_hidden_states)

    att_img = torch.ones([img_proj.shape[0], img_proj.shape[1]+1],
                         dtype=inputs.pixel_values.dtype,
                         device=device) 


    return att_img, img_proj


  def forward(self,text,image,target):



    self.tokenizer.padding_side = "right"

    att_img, img_proj = self.visual_encoder(image)

    tokenized_text = self.tokenizer(text, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_length,  add_special_tokens=False)
    tokenized_target = self.tokenizer(target, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_length,  add_special_tokens=False)
    text_embeds = self.adapter_model.model.get_input_embeddings()(tokenized_text.input_ids.to(device))


    self.tokenizer.bos_token_id = self.tokenizer.pad_token_id
    bos = torch.ones([tokenized_text.input_ids.shape[0], 1],
                         dtype=tokenized_text.input_ids.dtype,
                         device=device) * self.tokenizer.bos_token_id


    bos_embeds = self.adapter_model.model.get_input_embeddings()(bos)

    input_embeds = torch.cat([bos_embeds, img_proj, text_embeds.to(device)], dim=1)



    attention_mask = torch.cat([att_img.to(device), tokenized_text.attention_mask.to(device)],dim=1)



    targets = tokenized_target.input_ids.masked_fill(
                tokenized_target.input_ids == self.tokenizer.pad_token_id, -100
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





exp_name = "name"
exp_path = "path to model folder"


model_name = model_name = "google/flan-t5-large"
model = MultimodalLLM(model_name)

model.to(device)


lr = 0.00001


optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

import os, sys
from pytorchtools import EarlyStopping
accum_iter=4

# For cross entropy loss
def train_model(model, patience, n_epochs):
    epochs = n_epochs
#     clip = 5

  
    train_loss_list=[]
    val_loss_list=[]
    
        # initialize the experiment path
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    # initialize early_stopping object
    chk_file = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=chk_file)


    model.train()
    for i in range(epochs):
#         total_acc_train = 0

      
        #scheduler.step()

        total_loss_train = 0
        total_train = 0
        correct_train = 0

        for batch_idx,data in enumerate(dataloader_train):

          img_id = data["img_id"]
          text = data["text"]
          image = data["image"]
          target= data["target"]
          
          model.zero_grad()

          #print(image.shape)

          output = model(text = text, image = image,target=target)


          loss = output["loss"]
          loss=loss/accum_iter
          loss.backward()
          if((batch_idx)%accum_iter==0)or(batch_idx+1==len(dataloader_train)):
            optimizer.step()
            optimizer.zero_grad()
          total_loss_train += loss.item()
          total_train += image.size(0)

        
        train_loss = total_loss_train/total_train
        model.eval()

        total_loss_val = 0
        total_val = 0
        correct_val = 0

        with torch.no_grad():
            for data in dataloader_val:   

              img_id = data["img_id"]
              text = data["text"]
              image = data["image"]
              target=data["target"]
              
              model.zero_grad()
              output = model(text = text, image = image,target=target)
              val_loss = output["loss"]
              total_loss_val += val_loss.item()
              total_val += image.size(0)



     
        val_loss = total_loss_val/total_val

   
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        early_stopping(val_loss, model)
        
        print(f'Epoch {i+1}: train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}')
        model.train()
        torch.cuda.empty_cache()
        

    model.load_state_dict(torch.load(chk_file))

    
    return  model, train_loss_list, val_loss_list, i
        

n_epochs = 30
 
patience = 10
model, train_loss_list, val_loss_list, epoc_num = train_model(model, patience, n_epochs)
