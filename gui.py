import tkinter
import tkinter.ttk as ttk
from tkinter import Canvas, filedialog
from tkinter.constants import BOTH, CENTER
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import os, glob



window = tkinter.Tk()

style = ttk.Style(window)
style.theme_use('clam')
print(style.theme_use())

window.title('Horace')
window.geometry("550x500")

policefont = ('Yu Gothic', 16, 'bold')
subtitle = tkinter.Label(window,text='Ajouter photo', width=20, font=policefont, justify=CENTER)
uploadbutton = tkinter.Button(window, text='Charger une image', width=25, command =lambda:upload_image())
searchbutton = tkinter.Button(window,text="Trouver une correspondance", width=25, command=lambda:upload_image()) #changer fonction par celle pour comparer
subtitle2 = tkinter.Label(window, text="Correspondance", width=20, font=policefont, justify=CENTER)


subtitle.grid(row=1, column=1)
subtitle2.grid(row=1, column=2)
uploadbutton.grid(row=2, column=1, pady=10)
searchbutton.grid(row=3, column=1, pady=2)



def upload_image() :
    global img
    img_type = [('Jpg Files', '*.jpg')] #définition des formats photo
    img_name = filedialog.askopenfilename(filetypes=img_type) #ouverture du répertoire photo avec seulement les photos jpg

    img = ImageTk.PhotoImage(file=img_name)

    img2 = Image.open(img_name)
    #print("taille", img2.width, type(img2.width))

    img_sized = img2.resize((int((img2.width)/10), int((img2.height)/10)), Image.ANTIALIAS)


    img = ImageTk.PhotoImage(img_sized)
    display_img = tkinter.Button(window, image = img) #affichage de l'image en tant que bouton
    display_img.grid(row=4, column=1, pady=10, padx=10)


window.mainloop()
