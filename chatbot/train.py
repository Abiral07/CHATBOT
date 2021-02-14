import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader


from model import FFNeuralNet
with open('intents.json', 'r') as f:
    intents = json.load(f)
from nlp_fn import  tokenize, stem, bag_of_words

#empty list   
all_words = []
tags = []
xy = [] #holds both patterns and tags

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #1. tokenize each pattern
        w = tokenize(pattern)
        all_words.extend(w) #add all_words of a list to next list
        
        #add xy in the corpus#this will know thw pattern and corresponding tag
        xy.append((w, intent['tag']))
        #add documents(both patterns and tag)to end of list
        
        #add to our tags list
        if intent['tag'] not in tags:
            tags.append(intent['tag'])
    
# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

#print(all_words)
#print(tags)
#print(xy)

# create training data
X_train = [] #bow
y_train = [] #associated label for each tag

for(pattern_sentence, tag) in xy:
     # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels
    label = tags.index(tag)
    y_train.append(label)
    
#conversion to numpy array#x y to feature in a label 
X_train = np.array(X_train)
y_train = np.array(y_train)

#pytorch dataset
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    #acccess dataset with an index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
# Hyper-parameters 
batch_size = 8
input_size = len(all_words)#len(X_train[0])length of each bow we created same len as allwords
hidden_size = 8
output_size = len(tags)#NO. OF DIFFERENT CLASSES OR TAGS
learning_rate=0.001
num_epochs= 1000
#print(input_size, len(all_words))
#print(output_size, tags)

    
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                         )
# pylint: disable=E1101
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# pylint: enable=E1101
model = FFNeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()#loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
#Specify learning rate in hperparameters

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        # pylint: disable=E1101
        labels = labels.to(dtype=torch.long).to(device)
        # pylint: enable=E1101

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
    if (epoch+1) % 100 == 0:
        #in every 100stp we print current epoch and all epoch
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f'final loss: {loss.item():.4f}')
        
        
#save data        
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"output_size": output_size,
"hidden_size": hidden_size,
"all_words": all_words,
"tags": tags
}
#serialize and save it to file
FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')

