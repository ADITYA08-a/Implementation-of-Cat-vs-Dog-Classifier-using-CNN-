import math,torch,os,json,random
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
import onnx, onnxscript


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

class CNN(nn.Module):
        def __init__(self)-> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 10, 5,padding=1)
            self.batchnorm1 = nn.BatchNorm2d(10)
            #Conv2d(a, b,c) - a is the no of color channels the incoming image has
            #b is the no of filters or out channels
            #c is the kernel size .  c x c kernel is generated
            #Convolution is done for all 3 channels with the same kernel size
            self.conv2 = nn.Conv2d(10, 36 , 5,padding=1)
            self.batchnorm2 = nn.BatchNorm2d(36)
            self.conv3 = nn.Conv2d(36,48,5,padding=1)
            self.batchnorm3 = nn.BatchNorm2d(48)
            self.conv4 = nn.Conv2d(48,128,5,padding=1)
            self.batchnorm4 = nn.BatchNorm2d(128)
            self.conv5 = nn.Conv2d(128,20, 5, padding=1)
            self.batchnorm5 = nn.BatchNorm2d(20)
            self.m1 = nn.MaxPool2d(3, stride = 2)
            self.m2 = nn.MaxPool2d(3, stride = 2)
            self.m3 = nn.MaxPool2d(3, stride = 2)
            self.m4 = nn.MaxPool2d(3, stride = 2)
            self.m5 = nn.MaxPool2d(3, stride= 1)
            self.dropout = nn.Dropout(0.3)
            self.Linear_Layer = nn.LazyLinear(3)
            # Linear needs a fixed number of features before starting, LazyLinear can adapt to any number of features,
           
        def forward(self, x):
            X = F.gelu(self.batchnorm1(self.conv1(x)))
            Op1 = self.m1(X)
            X = F.gelu(self.batchnorm2(self.conv2(Op1)))
            Op2 = self.m2(X)
            X = F.gelu(self.batchnorm3(self.conv3(Op2)))
            Op3 = self.m3(X)
            X = F.gelu(self.batchnorm4(self.conv4(Op3)))
            Op4 = self.m4(X)
            X = F.gelu(self.batchnorm5(self.conv5(Op4)))
            Op5 = self.m5(X)
            Flattened_Op = torch.flatten(Op5, start_dim= 1)
            
            Flattened_Op = self.dropout(Flattened_Op)
            FINAL_OP = self.Linear_Layer(Flattened_Op)
            #Flattened_Op = self.dropout(Flattened_Op)
            return FINAL_OP
            
def calculate_loss(input_pred,target):
    loss = nn.BCEWithLogitsLoss()
    output2 = loss(input_pred, target )
    return output2


labels = ["Cat", "Dog","None"]
model = CNN()

model.to(device)
dummy_tensor_inp = torch.randn(1,3,480,320).to(device)
_ = model(dummy_tensor_inp)

if __name__ == "__main__":
    Dataset_Path1 = "./archive/cats_set"
    Dataset_Path2 = "./archive/dogs_set"
    Images = []
    print(f"PyTorch Version: {torch.__version__} | GPU Available: {torch.cuda.is_available()} | GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    convert_transform = v2.Compose([
        #v2.ToImage(),
        v2.RandomResizedCrop( size=(480,320), scale=(0.8,1.0) ),
        v2.RandomHorizontalFlip(p = 0.5),
        v2.RandomRotation(20),
        v2.RandomAffine( degrees=10, translate=(0.1,0.1), scale=(0.9,1.1)),
        v2.ToDtype(torch.float32, scale = True),
        v2.RandomPerspective( distortion_scale=0.2, p=0.5 ),
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
    #random.shuffle(Input_tensors)
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
    random.shuffle(Input_tensors)
    split_index = int(0.85 * len(Input_tensors))

    Train_Data = Input_tensors[:split_index]
    Test_Data = Input_tensors[split_index:]
    
    
    print("Length of Training Data", len(Train_Data))
    print("Length of Testing Data", len(Test_Data))
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
   

    #model = CNN()
    # if nn.LazyLinear is used, then a dummy forward pass has to be done first
    
    #optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    optimizer = optim.AdamW(model.parameters(), lr = 0.002, weight_decay= 1e-4)
    dataloader_batch = DataLoader(Input_tensors, batch_size=10, shuffle= True)
    dataset_instance = ProceesedDataset(start=0, end= len(Train_Data),data_array= Train_Data)
    dataloader_batch_new = DataLoader(dataset_instance, batch_size = 20, num_workers= 0)

    print("Training Starts")
    for epoch in range(20):
        Total_loss = 0
        Accuracy = 0
        Total_size = 0
        #Batch_size = 0
        for batch_inputs, batch_ground_truths in dataloader_batch_new:
            #Batch_size = len(Batch_size)
            batch_inputs = batch_inputs.to(device)
            batch_ground_truths = batch_ground_truths.to(device)
            optimizer.zero_grad()   
            outputs = model(batch_inputs)
            # dont use model.forward(batch) instead of model(batch) as this could disable the registered hooks , internal and external
            predictions = torch.argmax(outputs, dim=1)
            actual_ = torch.argmax(batch_ground_truths, dim = 1)
            Accuracy = Accuracy + (predictions == actual_).sum().item()

            Loss = calculate_loss(outputs, batch_ground_truths)
            Total_loss = Total_loss + Loss.item()
            Total_size = Total_size + batch_inputs.size(0)
            # using .item will take the raw value direcly and strip away the rest of the tensor
            # if .item() is not used, then it could cause a logical error and lead to out of memory error
            Loss.backward()
            optimizer.step()
        print("Training Accuracy", Accuracy/ Total_size)        
        print(f"Epoch :", epoch,"Loss :", Total_loss)

    input_tensor = torch.randn(20,3,480,320)
    processed_test_instance = ProceesedDataset(start=0, end = len(Test_Data), data_array= Test_Data)          
    dataloader_batch_test = DataLoader(processed_test_instance, batch_size= 10, shuffle= False)

    num_batches = 0
    print("Testing ")
    with torch.no_grad():
        Total_Testing_loss = 0
        no_of_images = 0
        Testing_Accuracy = 0
        for batch_inp, batch_gt in dataloader_batch_test:
            batch_gt = batch_gt.to(device)
            batch_inp = batch_inp.to(device)
            num_batches += 1
            outputs = model(batch_inp)
            Test_Loss = calculate_loss(outputs, batch_gt)
            Test_predictions = torch.argmax(outputs, dim=1)
            Actual_ = torch.argmax(batch_gt, dim = 1)
            Total_Testing_loss = Test_Loss.item() + Total_Testing_loss
            Testing_Accuracy += (Test_predictions == Actual_).sum().item()
            no_of_images = no_of_images + len(batch_inp)
            #Total_size = Total_size + batch_inputs.size(0)
        print("Testing Accuracy : ", 100 * Testing_Accuracy/no_of_images)
        print("Average Test Loss",Total_Testing_loss/num_batches)
    
    torch.save(
        model.state_dict(), "CNN.pth"
    )

    #Inference Loop
    """
    inference_loader = []
    predictions_list = {}
    with torch.no_grad():
        for images, filenames in inference_loader():
            output = model(images)
            predictions = torch.argmax(output,dim = 1)

            for filename, pred in zip(filenames, predictions):
                predictions_list[filename] = labels[pred.item()]

            with open("predictions.json","w") as f:
                json.dump(predictions_dict,f, indent= 5)

        

    torch.onnx.export(
        model,                       # model to export
        (input_tensor,),               # inputs of the model
        "my_model.onnx",             # filename of the ONNX model
        input_names = ["input"],     # Rename inputs for the ONNX model
        dynamo = True               # True or False to select the exporter
    )
    """