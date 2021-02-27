import random
import time
import json
import numpy
import torch
from model import FFNeuralNet
from nlp_fn import bag_of_words, tokenize

# pylint: disable=E1101
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# pylint: enable=E1101
                      
with open('intents.json','r', errors="ignore")as f:
    intents=json.load(f)
    
FILE="data.pth"
data=torch.load(FILE)

input_size=data["input_size"]
hidden_size=data["hidden_size"]
output_size=data["output_size"]
all_words=data["all_words"]
tags=data["tags"]
model_state=data["model_state"]

model = FFNeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

#create chat
from tkinter import *
root = Tk()
# w, h = root.winfo_screenwidth(), root.winfo_screenheight()
# root.geometry("%dx%d+0+0" % (w, 670))
root.geometry("900x600+10+10")
# root.attributes('-zoomed', True)
bot_name="BOT"
def send():
        sentence = e.get() 
        txt.insert(END,"\n"+"YOU: "+sentence)
        sentence = sentence.lower()
        if sentence =="bye" :
            txt.insert(END,"\n"+bot_name +": Visit Again :)"+"\n")
            time.sleep(1)
            root.quit()
        else:
            sentence = tokenize(sentence)
            X = bag_of_words(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            # pylint: disable=E1101
            X = torch.from_numpy(X) #row fn reurns numy array
            # pylint: enable=E1101

            output = model(X)
            # pylint: disable=E1101
            _, predicted = torch.max(output, dim=1)
            # pylint: enable=E1101
            tag = tags[predicted.item()]
            # pylint: disable=E1101
            probs = torch.softmax(output, dim=1)
            # pylint: enable=E1101
            prob = probs[0][predicted.item()]
            
            if prob.item() > 0.75 :   
                for intent in intents["intents"]:
                    if tag == intent["tag"]:
                        response = random.choice(intent ['responses'])
                        txt.insert(END,"\n"+ bot_name +": "+ response +"\n")
            else:
                txt.insert(END,"\n"+bot_name+": Sorry! I did'nt understand your problem."+"\n")
        e.delete(0,END)

txt = Text(root, width=170, height=35)
txt.grid(row=2,column=0,columnspan=2)
# txt.insert(END,"Hello there! I'm here to help you in case you need any First-Aid advice. Let's chat! (type 'bye' to exit)"+"\n")
e= Entry(root, width=200,bg="#9D00FF", fg="white", borderwidth=3)
send=Button(root,text="Send", command=send, width=20, bg="#24a0ed", fg="white").grid(row=0,column=0, rowspan=2) 
e.grid(row=0,column=1, rowspan=2)
root.title("CHATBOT")
root.mainloop()