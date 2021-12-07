import tkinter
import tkinter.ttk as ttk
from tkinter import Canvas, filedialog
from tkinter.constants import BOTH, CENTER
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import os, glob

import main

window = tkinter.Tk()
style = ttk.Style(window)
style.theme_use('clam')
print(style.theme_use())

window.title('Horace')
window.geometry("680x450")

policefont = ('Yu Gothic', 16, 'bold')
subtitle = tkinter.Label(window,text='Ajouter photo', width=20, font=policefont, justify=CENTER)
uploadbutton = tkinter.Button(window, text='Charger une image', width=25, command =lambda:upload_image())
searchbutton = tkinter.Button(window,text="➡️", width=3, command=lambda:match())
subtitle2 = tkinter.Label(window, text="Correspondance", width=20, font=policefont, justify=CENTER)

subtitle.grid(row=2, column=1)
subtitle2.grid(row=2, column=3)
uploadbutton.grid(row=3, column=1, pady=10)
searchbutton.grid(row=4, column=2, pady=10, padx=10)


img = None
img_name = None
display_img = None
display_img2 = None
img_memory = 0  #flag pour la superposition d'img
img_memory_match = 0  #flag pour la superposition d'img matché
def match():

    global img_memory_match, display_img2

    if img_name != None:
        match = main.main(img_name)  
        nom = str(match).lower()
        print(nom)
        display_match = tkinter.Label(window, text=match, width=20, font=policefont, justify=CENTER)
        display_match.grid(row=6, column=3, pady=10, padx=10)
        
        if img_memory_match >= 1 :      #Pour éviter la superposition d'image à comparer
            display_img2.destroy() 
        
        img = Image.open("affiche/" + nom + ".jpg")
        baseheight = 300
        hpercent = (baseheight / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(hpercent)))
        
        img_sized = img.resize((wsize, baseheight), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img_sized)  
        
        
        display_img2 = tkinter.Label(window, image = img) #affichage de l'image en tant que bouton
        display_img2.image = img
        display_img2.grid(row=4, column=3, pady=10, padx=10)
        img_memory_match += 1
        

        
        
    else:
        print('no image imported')

def upload_image() :
    
    global img, img_name, img_memory, display_img
    
    if img_memory >= 1 :      #Pour éviter la superposition d'image à comparer
            display_img.destroy()    
    img_type = [('Jpg Files', '*.jpg')] #définition des formats photo
    img_name = filedialog.askopenfilename(filetypes=img_type) #ouverture du répertoire photo avec seulement les photos jpg
    print(img_name)
    img = ImageTk.PhotoImage(file=img_name)

    img2 = Image.open(img_name)
    
    print(img2.size)
    baseheight = 300

    #wpercent = (basewidth / float(img2.size[1]))
    hpercent = (baseheight / float(img2.size[1]))
    wsize = int((float(img2.size[0]) * float(hpercent)))
    #print("taille", img2.width, type(img2.width))
    

    img_sized = img2.resize((wsize, baseheight), Image.ANTIALIAS)


    img = ImageTk.PhotoImage(img_sized)
    display_img = tkinter.Label(window, image = img, bg ="white") #affichage de l'image en tant que bouton
    display_img.image = img
    display_img.grid(row=4, column=1, pady=10, padx=10)
    img_memory += 1
    


window.mainloop()
