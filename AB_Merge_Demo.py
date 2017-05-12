#KERAS_BACKEND="theano"
import numpy as np
import word2vec_utils as w2v
import data
from data_utils import split_dataset 
import keras
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
import subprocess
import shlex
import tkinter as tk
from chat_constants import *
try:
    import ttk as ttk
    import ScrolledText
except ImportError:
    import tkinter.ttk as ttk
    import tkinter.scrolledtext as ScrolledText
import time

MODEL = 'AmergeB_len30_301dim_w2v_in_and_out.h5'
REINFORCE = False

class neural_chatbot():
    
    def __init__(self):
        #self.initialize()
        self.load_model()
        self.w2v_model = w2v.initialize()
        self.count=0
        self.out='1 9 1 @Hi'
        print( 'Loaded word_frequencies data from disk' )
        self.word_freqs = np.load('words_in_order_of_freq.npy') 
        self.embed_dim = EMBED_DIM
        self.data_dim = 1
        self.timesteps = 26
        self.vocab_size = 10000
        self.emot_size = 4
        self.max_sent_length = MAX_SENT_LENGTH
        self.reinforce = 'neutral'
    def save_model(self):
        # serialize model to JSON
        model_json = self.one_hot_chat_model.to_json()
        with open("one_hot_chat_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.one_hot_chat_model.save_weights("one_hot_chat_net.h5")
        print("Saved model to disk")
    
    def load_model(self):
        old_load = '''
        # load json and create model
        json_file = open('one_hot_chat_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.one_hot_chat_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.one_hot_chat_model.load_weights("one_hot_chat_net.h5")
        '''
        
        self.model = keras.models.load_model(MODEL)
        print("Loaded model from disk")
        
    
    def RateSentiment(self,sentiString):
        #open a subprocess using shlex to get the command line string into the correct args list format
        p = subprocess.Popen(shlex.split("/usr/bin/java -jar /home/paperspace/EmotionalChatBot/SentiStrength.jar stdin sentidata \
                                    /home/paperspace/EmotionalChatBot/SentStrength_Data_Sept2011/ "),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        #communicate via stdin the string to be rated. Note that all spaces are replaced with +
        stdout_text, stderr_text = p.communicate(sentiString.replace(" ","+"))
        #remove the tab spacing between the positive and negative ratings. e.g. 1    -5 -> 1-5
        stdout_text = stdout_text.rstrip().replace("\t","")
        return stdout_text   
    
            
    def reinforcement_learn(self,user_emotion,sys_response):
        total_emo = int(user_emotion[0]) - int(user_emotion[3])
        sys_res = np.argmax(sys_response,axis=2)
        adam = keras.optimizers.Adam(lr = 100)#default 0.001
        self.one_hot_chat_model.compile( optimizer=adam,loss = 'categorical_crossentropy',metrics = ['accuracy'])
        if total_emo > 0:
            #sys_res = (10*sys_res)+10
            sys_res_categorical=to_categorical(sys_res.astype(int),self.vocab_size+3).reshape((1,self.max_sent_length,self.vocab_size+3))
            self.reinforce='positive'
            self.one_hot_chat_model.fit(self.user_input,sys_res_categorical,epochs=3)
        else:
            sys_res_categorical=to_categorical(sys_res.astype(int),self.vocab_size+3).reshape((1,self.max_sent_length,self.vocab_size+3))*(-1)
            self.reinforce='negative'
            self.one_hot_chat_model.fit(self.user_input,sys_res_categorical,epochs=3)
        self.save_model()
        self.load_model()
        
    #def convert3d_to_4d(sys_response3d):
    #    sys_res3d_standardized = (10*sys_res)+10
    #    sys_response = to_categorical(sys_res3d_standardized.astype(int),20).reshape((1,300,-1,20))

    def model_response(self,User_B):
        #User_A1 = "5 5 1 @I really love you. I think I want to marry you!"
        #User_B = "1 5 5 @yeah well I hate you and never want to see you again! "
        #User_A1 = w2v.vectorize( w2v_model, User_A1 , pad_length = max_sent_length)
        Sen_User_B = self.RateSentiment(User_B)
        print(Sen_User_B)
        self.count += 1
        if REINFORCE and self.count>1:
            self.reinforcement_learn(Sen_User_B,self.sys_prediction)
        
        User_B = Sen_User_B[0]+" "+str(11-int(Sen_User_B[0])-int(Sen_User_B[3]))+" "+Sen_User_B[3]+" "+EMOTE_DELIMITER+User_B
        self.User_B_in  = w2v.vectorize( User_B, pad_length = MAX_SENT_LENGTH,model=self.w2v_model ) 
        User_A = w2v.vectorize( self.out, pad_length = MAX_SENT_LENGTH,model=self.w2v_model )
        
        #Merge the prev and the existing sentences together
        self.user_input = [ np.array( [User_A]).reshape(1,MAX_SENT_LENGTH,EMBED_DIM), np.array([self.User_B_in]).reshape(1,MAX_SENT_LENGTH,EMBED_DIM) ]
        self.sys_prediction = self.model.predict(self.user_input)[0]
        print(self.sys_prediction.shape)

        self.out = w2v.unvectorize_sentence( self.sys_prediction )
        print(self.out)    
        return (self.out)
        #sys_pred = np.argmax(prediction, axis=2)
        #self.sys_prediction = (sys_pred-10)/10
        #return self.most_likely_sent( self.sys_prediction[0], metadata['w2idx'].keys(),self.w2v_model)
    
class TkinterGUIExample(tk.Tk):

    def __init__(self, *args, **kwargs):
        """
        Create & set window variables.
        """
        tk.Tk.__init__(self, *args, **kwargs)

        self.title("Di-Feelbot")
        self.neuralbot = neural_chatbot()
        self.initialize()

    def initialize(self):
        """
        Set window layout.
        """
        self.grid()

        self.respond = ttk.Button(self, text='Get Response', command=self.get_response)
        self.respond.grid(column=0, row=0, sticky='nesw', padx=3, pady=3)

        self.usr_input = ttk.Entry(self, state='normal')
        self.usr_input.grid(column=1, row=0, sticky='nesw', padx=3, pady=3)

        self.conversation_lbl = ttk.Label(self, anchor=tk.E, text='Conversation:')
        self.conversation_lbl.grid(column=0, row=5, sticky='nesw', padx=3, pady=3)

        self.conversation = ScrolledText.ScrolledText(self, state='disabled')
        self.conversation.grid(column=0, row=6, columnspan=2, sticky='nesw', padx=3, pady=3)
        
        self.emo_lbl = ttk.Label(self, text='Sys emotion')
        self.emo_lbl.grid(column=0, row=1, sticky='nesw', padx=3, pady=3)
        
        self.emoPosvalue_lbl = ttk.Label(self, text='Pos emotion',foreground='blue')
        self.emoPosvalue_lbl.grid(column=0, row=2, sticky='nesw', padx=3, pady=3)
        
        self.emoNegvalue_lbl = ttk.Label(self, text='Neg emotion',foreground='red')
        self.emoNegvalue_lbl.grid(column=0, row=3, sticky='nesw', padx=3, pady=3)
        
        self.emoNeuvalue_lbl = ttk.Label(self, text='Neu emotion',foreground='black')
        self.emoNeuvalue_lbl.grid(column=0, row=4, sticky='nesw', padx=3, pady=3)
        
        self.uemo_lbl = ttk.Label(self, text='User emotion')
        self.uemo_lbl.grid(column=1, row=1, sticky='nesw', padx=3, pady=3)
        
        self.uemoPosvalue_lbl = ttk.Label(self, text='Pos emotion',foreground='blue')
        self.uemoPosvalue_lbl.grid(column=1, row=2, sticky='nesw', padx=3, pady=3)
        
        self.uemoNegvalue_lbl = ttk.Label(self, text='Neg emotion',foreground='red')
        self.uemoNegvalue_lbl.grid(column=1, row=3, sticky='nesw', padx=3, pady=3)
        
        self.uemoNeuvalue_lbl = ttk.Label(self, text='Neu emotion',foreground='black')
        self.uemoNeuvalue_lbl.grid(column=1, row=4, sticky='nesw', padx=3, pady=3)
        
        self.reinforce_lbl = ttk.Label(self, text='Reinforcement',foreground='red')
        self.reinforce_lbl.grid(column=2, row=1, sticky='nesw', padx=3, pady=3)
    
        
    def get_response(self):
        """
        Get a response from the chatbot and display it.
        """
        user_input = self.usr_input.get()
        self.usr_input.delete(0, tk.END)

        response = self.neuralbot.model_response(user_input)
        x=self.neuralbot.RateSentiment(user_input)
        self.emoPosvalue_lbl['text'] = response[0]
        self.emoNegvalue_lbl['text'] = response[4]
        self.emoNeuvalue_lbl['text'] = response[2]
        self.uemoPosvalue_lbl['text'] = x[0]
        self.uemoNegvalue_lbl['text'] = x[3]
        self.uemoNeuvalue_lbl['text'] = str(11-int(x[0])-int(x[3]))
        self.reinforce_lbl['text'] = self.neuralbot.reinforce + ' Reinforcement'
        self.conversation['state'] = 'normal'
        self.conversation.insert(
            tk.END, "Human: " + user_input + "\n" + "ChatBot: " + str(response[7:]) + "\n"
        )
        self.conversation['state'] = 'disabled'

        time.sleep(0.5)

w2v.unvectorize_initialize()
gui_example = TkinterGUIExample()
gui_example.mainloop()