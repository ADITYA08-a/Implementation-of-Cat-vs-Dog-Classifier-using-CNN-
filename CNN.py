import math,numpy,torch,os,json
import torch.nn as nn
import torch.utils as utils
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torch.utils.data import default_collate
import torchvision.io
import onn, onnxscript
 
if __name__ == "__main__":
    labels = ["Cat","Dog","None"]
    Dataset_Path1 = "./archive/cats_set"
    Dataset_Path2 = "./archive/dogs_set"
    Images = []
    print(f"PyTorch Version: {torch.__version__} | GPU Available: {torch.cuda.is_available()} | GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    convert_transform = v2.Compose([
        #v2.ToImage(),
        v2.Resize((52,30)),
        v2.RandomHorizontalFlip(p = 0.5),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean= [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    Input_tensors = []
    for file in os.listdir(Dataset_Path1):
        if file.split('.')[-1].lower() in ['png','jpg','jpeg','bmp']:
        #if file.lower().endswith('png','jpg','jpeg','bmp'):
            img_path = os.path.join(Dataset_Path1, file)    
            img = torchvision.io.decode_image(img_path)

            if img is not None:
                actual_tensor = convert_transform(img)
                if "cat" in file.lower():
                    target_tensor = torch.tensor([1.0,0.0,0.0], dtype = torch.float32)
                elif "dog" in file.lower():
                    target_tensor = torch.tensor([0.0,1.0,0.0], dtype = torch.float32)
                else:
                    target_tensor = torch.tensor([0.0, 0.0, 1.0], dtype = torch.float32)
                Input_tensors.append((actual_tensor, target_tensor))

    for file in os.listdir(Dataset_Path2):
        if file.split('.')[-1].lower() in ['png','jpg','jpeg','bmp']:
        #if file.lower().endswith('png','jpg','jpeg','bmp'):
            img_path = os.path.join(Dataset_Path2, file)
            img = torchvision.io.decode_image(img_path)
            if img is not None:
                actual_tensor = convert_transform(img)
                if "cat" in file.lower():
                    target_tensor = torch.tensor([1.0,0.0,0.0], dtype = torch.float32)
                elif "dog" in file.lower():
                    target_tensor = torch.tensor([0.0,1.0,0.0], dtype = torch.float32)
                else:
                    target_tensor = torch.tensor([0.0, 0.0, 1.0], dtype = torch.float32)
                Input_tensors.append((actual_tensor, target_tensor))

    length_of_train_dataset = math.ceil(0.9 * len(Input_tensors))
    length_of_test_dataset = math.ceil(0.1 * len(Input_tensors))
    Test_Dataset_Path = "./test1/test1"
    Testing_Input_Tensors = []
    predictions_dict = {}
    for file in os.listdir(Test_Dataset_Path):
        if file.split('.')[-1].lower() in ['png', 'jpg','jpeg','bmp']:
            test_image_path = os.path.join(Test_Dataset_Path, file)
            test_image = torchvision.io.decode_image(test_image_path)
            if test_image is not None:
                actual_test_tensor = convert_transform(test_image)
                if "cat" in file.lower():
                    target_tensor = torch.tensor([1.0,0.0,0.0], dtype= torch.float32)
                elif "dog" in file.lower():
                    target_tensor = torch.tensor([0.0,1.0, 0.0], dtype = torch.float32)
                else:
                    target_tensor = torch.tensor([0.0,0.0, 1.0], dtype = torch.float32)
                Testing_Input_Tensors.append((actual_test_tensor,target_tensor))
                
    class ProceesedDataset(torch.utils.data.IterableDataset):
        def __init__(self,start,end,data_array):
            super(ProceesedDataset).__init__()
            self.start = start
            self.end = end
            self.data_array = data_array
        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                iter_start = self.start
                iter_end = self.end
            else:
                per_worker = int(math.ceil((self.end - self.start)/worker_info.num_workers))
                worker_id = worker_info.id
                iter_start = self.start + worker_id * per_worker
                iter_end = min(iter_start + per_worker, self.end)
            
            for index in range(iter_start,iter_end):
                yield self.data_array[index]
            #return iter(range(iter_start, iter_end))
            
    class CNN(nn.Module):
        def __init__(self)-> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 30, 5)
            #Conv2d(a, b,c) - a is the no of color channels the incoming image has
            #b is the no of filters or out channels
            #c is the kernel size .  c x c kernel is generated
            #Convolution is done for all 3 channels with the same kernel size
            self.conv2 = nn.Conv2d(30, 20 , 5)
            self.m1 = nn.MaxPool2d(3, stride = 2)
            self.m2 = nn.MaxPool2d(3, stride = 2)
            self.Linear_Layer = nn.LazyLinear(3)
            # Linear needs a fixed number of features before starting, LazyLinear can adapt to any number of features,
           
        def forward(self, x):
            X = F.relu(self.conv1(x))
            Op1 = self.m1(X)
            X = F.relu(self.conv2(Op1))
            Op2 = self.m2(X)
            Flattened_Op = torch.flatten(Op2, start_dim= 1)
            FINAL_OP = self.Linear_Layer(Flattened_Op)
            return FINAL_OP
            
    def calculate_loss(input_pred,target):
        loss = nn.BCEWithLogitsLoss()
        output2 = loss(input_pred, target )
        return output2

    model = CNN()
    # if nn.LazyLinear is used, then a dummy forward pass has to be done first
    dummy_tensor_inp = torch.randn(1,3,52,30)
    _ = model(dummy_tensor_inp)
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    dataloader_batch = DataLoader(Input_tensors, batch_size=10, shuffle= True)
    dataset_instance = ProceesedDataset(start=0, end= len(Input_tensors),data_array= Input_tensors)
    dataloader_batch_new = DataLoader(dataset_instance, batch_size = 20, num_workers= 0)

    print("Training Starts")
    for epoch in range(20):
        Total_loss = 0
        for batch_inputs, batch_ground_truths in dataloader_batch_new:
            optimizer.zero_grad()   
            outputs = model(batch_inputs)
            # dont use model.forward(batch) instead of model(batch) as this could disable the registered hooks , internal and external
            Loss = calculate_loss(outputs, batch_ground_truths)
            Total_loss = Total_loss + Loss.item()
            # using .item will take the raw value direcly and strip away the rest of the tensor
            # if .item() is not used, then it could cause a logical error and lead to out of memory error
            Loss.backward()
            optimizer.step()        
        print(f"Epoch :", epoch,"Loss :", Total_loss)

    input_tensor = torch.randn(20,3,52,30)
    processed_test_instance = ProceesedDataset(start=0, end = len(Testing_Input_Tensors), data_array= Testing_Input_Tensors)          
    dataloader_batch_test = DataLoader(processed_test_instance, batch_size= 10, shuffle= False)

    print("Testing ")
    with torch.no_grad():
        Total_Testing_loss = 0
        no_of_images = 0
        for batch_inp, batch_gt in dataloader_batch_test:
            outputs = model(batch_inp)
            Test_Loss = calculate_loss(outputs, batch_gt)
            Total_Testing_loss = Test_Loss.item() + Total_Testing_loss
            no_of_images = no_of_images + len(batch_inp)
        print("Average Test Loss",Total_Testing_loss/no_of_images)

    torch.onnx.export(
        model,                       # model to export
        (input_tensor,),               # inputs of the model
        "my_model.onnx",             # filename of the ONNX model
        input_names = ["input"],     # Rename inputs for the ONNX model
        dynamo = True               # True or False to select the exporter
    )