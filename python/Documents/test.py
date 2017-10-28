import tkinter
import os
from tkinter import messagebox

top = tkinter.Tk()

x=""
def helloCallBack():
  messagebox.showinfo( "Hello Python", "Hello World")


def training():
  os.system('python train2.py')

def rec():
  os.system('python facerec2.py')





B = tkinter.Button(top, text ="Train", command = training)
C = tkinter.Button(top, text ="Recognize", command = rec)

B.pack()
C.pack()
top.mainloop()
