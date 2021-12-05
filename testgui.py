
from tkinter import *  
from PIL import ImageTk,Image  
root = Tk()  
canvas = Canvas(root, width = 300, height = 300)  
canvas.pack()  

img = Image.open("affiche/oak.jpg")
img_sized = img.resize((int((img.width)/10), int((img.height)/10)), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img_sized)  
canvas.create_image(20, 20, anchor=NW, image=img) 
root.mainloop() 