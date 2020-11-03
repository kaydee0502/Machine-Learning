import pandas as pd
import numpy as np
import tkinter

from tkinter import Frame,Canvas,Tk,Listbox,Scrollbar,END
import tkinter.font as tkFont
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



win = tkinter.Tk()
win.geometry("800x600")
win.title('XitiZ')
win.resizable(False, False)
st = tkinter.StringVar()
it = tkinter.StringVar()
head = tkFont.Font(family="Lucida Grande", size=20)

can = tkinter.Canvas(win, height=40, width=400,bg = 'teal')
can.place(x = 175, y = 58)
tkinter.Label(win,font=head,text = "Enter a movie name to find similar movies :-").place(x = 10,y = 10)
val = tkinter.Entry(win,textvariable = st, width=58)
val.place(x = 200, y = 70)
tkinter.Label(win,font=head,text = "How many suggestions do you want :-").place(x = 10,y = 110)

can2 = tkinter.Canvas(win, height=40, width=400,bg = 'teal')
can2.place(x = 175, y = 150)
val2 = tkinter.Entry(win,textvariable = it, width=58)
val2.place(x = 200, y = 162)
button = tkinter.Button(win, text='Search',width=25,command = lambda:[display()])
button.place(x = 275, y = 205)





frame = Frame(win)
frame.place(x = 25, y = 240)

listNodes = Listbox(frame, width=80, height=20, font=("Helvetica", 12))
listNodes.pack(side="left", fill="y")

scrollbar = Scrollbar(frame, orient="vertical")
scrollbar.config(command=listNodes.yview)
scrollbar.pack(side="right", fill="y")

listNodes.config(yscrollcommand=scrollbar.set)


    
win.mainloop()
###### helper functions. Use them when needed #######

##################################################

##Step 1: Read CSV File
df = pd.read_csv("movie_dataset.csv")
##Step 2: Select Features
features = ["keywords","cast","genres","director"]
##Step 3: Create a column in DF which combines all selected features
for i in features:
    df[i] = df[i].fillna('')


def combine_features(x):
    return x["keywords"] +" "+ x["cast"] +" "+ x["genres"] +" "+ x["director"]

df["combined_feature"] = df.apply(combine_features,axis=1)

##Step 4: Create count matrix from this new combined column
vector = CountVectorizer()
vector_fit = vector.fit_transform(df["combined_feature"])
##Step 5: Compute the Cosine Similarity based on the count_matrix

cos_sim = cosine_similarity(vector_fit)


def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]


def display():
    listNodes.delete(0,'end')
    s = r"{}".format(st.get())
    n = int(r"{}".format(it.get()))
    i = 0
    movie_user_likes = s

    movie_index = get_index_from_title(movie_user_likes)

    similiar_movies = list(enumerate(cos_sim[movie_index]))
    sorted_similiar_movies = sorted(similiar_movies,key= lambda x:x[1], reverse = True)
    listNodes.insert(END, str("\u2022 If you enjoyed :-"))
    listNodes.insert(END, str(get_title_from_index(sorted_similiar_movies[i][0])))
    listNodes.insert(END, str(""))
    listNodes.insert(END, str("\u2022 Then you might like :-"))
    


    while i < n:
        listNodes.insert(END, str(get_title_from_index(sorted_similiar_movies[i+1][0])))
        i += 1





