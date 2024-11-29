import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import AutoTokenizer
import os
# Check if CUDA is available


from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    print('local ddp',ddp_local_rank)
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"



model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"


model = AutoModelForImageTextToText.from_pretrained( model_id,torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(model_id,padding_side='left')
tokenizer=AutoTokenizer.from_pretrained(model_id,padding_side='left')
from peft import LoraConfig,get_peft_model

lora_config=LoraConfig(
    task_type='CAUSAL_LM',
    r=64,
    lora_alpha=16,
    use_rslora=True,
    lora_dropout=0.1,
    target_modules=['q_proj','v_proj']

)
model=get_peft_model(model,lora_config)
model.print_trainable_parameters()

# if master_process:
#     for name,param in model.named_parameters():
#         if param.requires_grad :
#             print(name)





import json
with open('llama_stuff/Finetuning_images.json','r')as f:
    image_paths=json.load(f)
with open('llama_stuff/Finetuning_captions.json','r')as f:
    answers=json.load(f)
with open('llama_stuff/templates.json','r')as f:
    prompts=json.load(f)

image_paths=['./llama_stuff/2nd_exp/2nd_FT_images/'+img for img in image_paths]


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
max_length=0
class CustomDataset(Dataset):
    def __init__(self, prompts, answers, image_paths, processor, tokenizer):
        self.prompts = processor.apply_chat_template(prompts, continue_final_message=True, tokenize=False)
        self.answers = answers
        self.image_paths = image_paths
        self.processor = processor
        self.tokenizer = tokenizer
        
        self.independant_sequences = [prompt + answer +tokenizer.eos_token for prompt, answer in zip(self.prompts, self.answers)]
        self.independant_images = [Image.open(path) for path in image_paths]
        self.all_data = self._prepare_data()

    def _prepare_data(self):
        
        all_images,inputs,processed_inputs,unprocessed_labels,words_to_predict,processed_labels=[],[],[],[],[],[]
        all_data=[]
        word_to_predict=None

        for image , independant_sequence,answer in zip(self.independant_images,self.independant_sequences,self.answers):
            nbr_answer_tokens=len(tokenizer.tokenize(answer))
            
            
            same_seq_images=[image for _ in range(0,nbr_answer_tokens)]
            all_images.extend(same_seq_images)
            input=independant_sequence
            for _ in range(0,nbr_answer_tokens):
                word_to_predict=tokenizer.decode(tokenizer.encode(input)[-1],padding_side='left') 
                input=tokenizer.decode(tokenizer.encode(input)[: -1],padding_side='left')
                inputs.append(input)

                tokenized_label=tokenizer(word_to_predict,add_special_tokens=False,return_tensors="pt",padding_side='left',padding='max_length',max_length=max([len(tokenizer.encode(i)) for i in self.independant_sequences]))['input_ids']
                unprocessed_labels.append(torch.where(tokenized_label != 128004,tokenized_label,-100))
            
            
            processed_labels = torch.cat(unprocessed_labels, dim=0)
            #print(processed_labels)


        for image,input_seq,label in zip(all_images,inputs,processed_labels):
            all_data.append({
                        'image': image,
                        'input_seq': input_seq,
                        'label': label  # Remove extra dimension
                    })
        return all_data
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        item = self.all_data[idx]
        return item['image'], item['input_seq'], item['label']
    



# Collate function for batch processing
def collate_fn(batch):
    images, inputs, labels = zip(*batch)
    # Process images and inputs using the processor
    processed_inputs = processor(
        images, inputs, 
        return_tensors="pt", 
        add_special_tokens=False, 
        padding='max_length', 
        max_length=max(label.size(0) for label in labels), 
        padding_side='left'
    )
     # Stack labels into a tensor
    stacked_labels = torch.stack(labels)
    return processed_inputs, stacked_labels

# Dataset and DataLoader setup
dataset = CustomDataset(prompts, answers, image_paths, processor, tokenizer)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)



from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm


optimizer=AdamW(model.parameters(),lr=1e-4,weight_decay=0.01)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model


num_epochs = 3
num_training_steps = num_epochs * len(data_loader)
lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

progress_bar = tqdm(range(num_training_steps))
def calculate_loss(logits, labels):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    cross_entropy_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    return cross_entropy_loss
model.train()


          
counter=0
model.train()
optimizer.zero_grad()
for epoch in range(num_epochs):
    loss_accum = torch.zeros(1,1).to(device)
    for batch_idx, (processed_inputs, batch_labels) in enumerate(data_loader):
        if   (ddp_local_rank*len(data_loader))//ddp_world_size  <= batch_idx < (ddp_local_rank+1)*len(data_loader)//ddp_world_size:
            batch_labels = batch_labels.to(device)
            marco=processed_inputs.to(device)
            # added after video, this field is also used by the forward pass.
            if ddp:
                model.require_backward_grad_sync = True
            outputs = model(**marco)
            loss=calculate_loss(outputs.logits,batch_labels.to(device)).mean()
            loss.backward()
            loss_accum += loss.detach().to(device)
            
            if ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if device_type == "cuda":
                    torch.cuda.synchronize() # wait for the GPU to finish work

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            del marco
            del batch_labels 
            counter+=1
            print(f'counter  : {counter}/{num_epochs*len(data_loader)}')
            if counter == len(data_loader)//(ddp_world_size):
            
                if master_process:
                    save_path = "lora_checkpoint_1epoch"
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        raw_model = model.module
                    else:
                        raw_model = model

                    # Save only the LoRA parameters
                    print(f'gpu {ddp_local_rank} saved lora checkpoint 1 epoch')
                    raw_model.save_pretrained(save_path)

            if counter == 2*len(data_loader)//(ddp_world_size):
            
                if master_process:
                    save_path = "lora_checkpoint_2epoch"
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        raw_model = model.module
                    else:
                        raw_model = model

                    # Save only the LoRA parameters
                    print(f'gpu {ddp_local_rank} saved lora checkpoint 2 epochs')
                    raw_model.save_pretrained(save_path)
            

            # if master_process:
            #     print(f"device : {device}   | loss: {loss.item():.6f}  | epoch {epoch}/{num_epochs} |  batch : {batch_idx}")
        
            
if master_process:
    save_path = "lora_checkpoint"
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        raw_model = model.module
    else:
        raw_model = model

    # Save only the LoRA parameters
    print(f'gpu {ddp_local_rank} saved lora checkpoint')
    raw_model.save_pretrained(save_path)

if device_type == "cuda":
        print(f'gpu {ddp_local_rank} waiting')
        torch.cuda.synchronize()


if ddp:
    destroy_process_group()
    torch.cuda.empty_cache() 
    print(f'gpu {ddp_local_rank} exited')

