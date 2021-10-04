from tkinter import *
from PIL import ImageTk, Image



root = Tk()
root.title("ciaobella")
root.geometry("400x400")








































# import threading
# threading.Thread(target=five_seconds).start # in command






































# from tkinter import ttk
# progr = ttk.Progressbar(root, orient=HORIZONTAL, 
#                         length=300, mode="determinate")
# progr.pack(pady=20)

# def step():
#     progr["value"] += 1
#     # progr.start()
#     # progr.stop()
    

# but = Button(root, text="progress", command=step)
# but.pack(pady=20)































# can = Canvas(root, width=200, height=200, bg="white")
# can.pack()

# img = PhotoImage(file="")


































# def number():
#     try:
#         print(int(my_box.get()))
#     except ValueError:
#         raise Exception("error")
#     pass

# my_label = Label(root, text="number")
# my_label.pack(pady=20)

# my_box = Entry(root)
# my_box.pack(pady=10)

# but = Button(root, text="enter", command=number)
# but.pack(pady=5)

# answer = Label(root, text="")
# answer.pack(pady=20)



























# from tkinter import ttk

# my_notebook = ttk.Notebook(root)
# my_notebook.pack(pady=15)

# my_frame1 = Frame(my_notebook, width=500, height=500, bg="blue")
# my_frame2 = Frame(my_notebook, width=500, height=500, bg="red")
# my_frame1.pack(fill="both", expand=1)
# my_frame2.pack(fill="both", expand=1)

# my_notebook.add(my_frame1, text="frame1")
# my_notebook.add(my_frame2, text="frame2")





























# my_frame = Frame(root)
# scroll = Scrollbar(my_frame, orient=VERTICAL)

# my_listbox = Listbox(my_frame, width=50, yscrollcommand=scroll.set, selectmode=EXTENDED)
## print(my_listbox.curselection()) # list indeces

# scroll.config(command=my_listbox.yview)
# scroll.pack(side=RIGHT, fill=Y)

# my_listbox.pack()
# my_frame.pack()

# my_list = ["bla1", "bla2", "bla3", "bla1", "bla2", "bla3", "bla1", "bla2",
#            "bla3","bla1", "bla2", "bla3", "bla1", "bla2", "bla3", "bla1", 
#            "bla2", "bla3"]
# for item in my_list:
#     my_listbox.insert(END, item)





































# my_listbox = Listbox(root)
# my_listbox.pack()

# my_listbox.insert(END, "bla1")
# my_listbox.insert(0, "bla2")

# my_list = ["bla1", "bla2", "bla3"]
# for item in my_list:
#     my_listbox.insert(END, item)


# def delete():
#     my_listbox.delete(ANCHOR)

# def select():
#     lab.config(text=my_listbox.get(ANCHOR))

# my_botton = Button(root, text="Delete", command=delete)
# my_botton.pack()

# my_botton = Button(root, text="select", command=select)
# my_botton.pack()

# lab = Label(root, text="")
# lab.pack()






















# def hide_all_frames():
#     for widget in file_new_fram.winfo_children():
#         widget.destroy()




































# # panels
# panel_1 = PanedWindow(bd=4, relief="raised", bg="red")
# panel_1.pack(fill=BOTH, expand=1)
# left_label = Label(panel_1, text="bla")

# panel_2 = PanedWindow(panel_1, orient=VERTICAL, bd=4, relief="raised", bg="red")
# panel_1.add(panel_2)
# left_label = Label(panel_1, text="bla")

































# def our_command():
#     pass

# def file_new():
#     file_new_frame.pack(fill="both", expand=1)


# my_menu = Menu(root)
# root.config(menu=my_menu)

# file_menu = Menu(my_menu)
# my_menu.add_cascade(label="File", menu=file_menu)
# file_menu.add_command(label="New", command=file_new)
# file_menu.add_separator()
# file_menu.add_command(label="Exit", command=root.quit)

# edit_menu = Menu(my_menu)
# my_menu.add_cascade(label="Edit", menu=edit_menu)
# edit_menu.add_command(label="Cut", command=our_command)
# edit_menu.add_separator()
# edit_menu.add_command(label="Copy", command=our_command)

# opt_menu = Menu(my_menu)
# my_menu.add_cascade(label="Options", menu=opt_menu)
# opt_menu.add_command(label="Cut", command=our_command)
# opt_menu.add_separator()
# opt_menu.add_command(label="Copy", command=our_command)



# file_new_frame = Frame(root, width=400, height=400, bg="red")


























# def selected(event):
#     print(event)

# options = ["bla1", "bla2", "bla3"]

# clicked = StringVar()
# clicked.set(options[0])

# drop = OptionMenu(root, clicked, *options, command=selected)
# drop.pack()

# from tkinter import ttk

# mycombo = ttk.Combobox(root, value=options)
# mycombo.current(0)
# mycombo.bind("<<ComboboxSelected>>", selected)
# mycombo.pack()

























# def clicker(event):
#     print(event)

# myButton = Button(root, text="Click")
# myButton.bind("<Button-3>", clicker)
# myButton.pack()
































# class Elder():
    
#     def __init__(self, master):
#         myFrame = Frame(master)
#         myFrame.pack()
#         self.myButton = Button(master, text="bla", command=self.clicker)
#         self.myButton.pack()
    
    
#     def clicker(self):
#         print("blablabla")


# e = Elder(root)





















# # Remove Labels in pack
# lbl.pack_forget() # hide
# lbl.destroy()     # completely destroy
# print(myButton.winfo_exists())

# # Remove Labels in grid
# lbl.grid_forget()
# lbl.destroy()
# print(myButton.winfo_exists())


















# # dropdown box

# var = StringVar()
# var.set("bla1") # default
# options = ["bla1", "bla2", "bla3", "bla4"]

# drop = OptionMenu(root, var, *options)
# drop.pack()

# var.get()
















# var = IntVar()

# c = Checkbutton(root, text="Box", variable=var, )
# c.pack()

# var.get() # 0 unchecked 1 checked
























# vertical = Scale(root, from_=0, to=100)
# vertical.pack()

# horizontal = Scale(root, from_=0, to=100, orient=HORIZONTAL)
# horizontal.pack()

# horizontal.get()
# vertical.get()




# def slide(var):
#     my_label = Label(root, text=vertical.get()).pack()
#     root.geometry(str(vertical.get())+"x400")

# vertical = Scale(root, from_=0, to=100, command=slide)
# vertical.pack()











# from tkinter import filedialog
# root.filename = filedialog.askopenfilename(initialdir="/",
#                                            title="Select a file",
#                                            filetypes=(("png files", "*.png"),
#                                                       ("all files", "*.*")))















# def open():
#     top = Toplevel()
#     lbl = Label(top, text="blabla").pack()
#     global variable

# btn = Button(root, text="second", command=open).pack()













# frame = LabelFrame(root, text="This is frame", padx=5, pady=5)
# frame.pack(padx=10, pady=10)















# my_img = ImageTk.PhotoImage(Image.open("_117796291_gettyimages-962098266.jpg"))
# my_label = Label(image=my_img)
# my_label.pack()







# from tkinter import messagebox
# response = messagebox.askquestion("blabla", "question") # showinfo, showwarning, showerror, askquestion, askokcancel, askyesno










# MODES = [
#     ("Pepperoni","Pepperoni"),
#     ("bla","bla"),
#     ("asjbnd","asjbnd"),
#     ("lksajdb","lksajdb"),
#     ]




# def clicked(value):
#     b = Button(frame, text=r.get(), padx=50, pady=50)
#     b.pack()

# r = StringVar()
# r.set("Pepperoni")
# for text, mode in MODES:
#     rb = Radiobutton(frame, text=text, variable=r, value=mode, command=lambda: clicked(r.get()))
#     rb.pack()
# rb2 = Radiobutton(frame, text="op2", variable=r, value=2, command=lambda: clicked(r.get()))
# rb2.pack()


# r = IntVar()
# rb1 = Radiobutton(frame, text="op1", variable=r, value=1, command=lambda: clicked(r.get()))
# rb1.pack()
# rb2 = Radiobutton(frame, text="op2", variable=r, value=2, command=lambda: clicked(r.get()))
# rb2.pack()







# b = Button(frame, text=r.get(), padx=50, pady=50)
# b.pack()




# e = Entry(root, width=35, borderwidth=5) # input data from user
# e.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
# # e.insert(0, "strunz")



















# # def my_click():
# #     my_label1 = Label(root, text="caz "+e.get())
# #     # my_label1.pack()
# #     my_label1.grid(row=0, column=0)


# # my_butt = Button(root, text="bla", padx=50, pady=50,
# #                  command=my_click) # state=DISABLED, 
# # my_butt.pack()

# def button_click(number):
#     current = e.get()
#     e.delete(0, END)
#     e.insert(0, str(current) + str(number))
#     return


# def button_clear():
#     e.delete(0, END)
#     return


# def button_add():
#     num1 = e.get()
#     global f_num
#     f_num = int(num1)
#     e.delete(0,END)
#     return


# def button_eq():
#     num2 = e.get()
#     e.delete(0, END)
#     e.insert(0, f_num+int(num2))
#     return



# but1 = Button(root, text="1", padx=40, pady=20, command=lambda: button_click(1))
# but2 = Button(root, text="2", padx=40, pady=20, command=lambda: button_click(2))
# but3 = Button(root, text="3", padx=40, pady=20, command=lambda: button_click(3))
# but4 = Button(root, text="4", padx=40, pady=20, command=lambda: button_click(4))
# but5 = Button(root, text="5", padx=40, pady=20, command=lambda: button_click(5))
# but6 = Button(root, text="6", padx=40, pady=20, command=lambda: button_click(6))
# but7 = Button(root, text="7", padx=40, pady=20, command=lambda: button_click(7))
# but8 = Button(root, text="8", padx=40, pady=20, command=lambda: button_click(8))
# but9 = Button(root, text="9", padx=40, pady=20, command=lambda: button_click(9))
# but0 = Button(root, text="0", padx=40, pady=20, command=lambda: button_click(0))

# but_add = Button(root, text="+", padx=39, pady=20, command=button_add)
# but_eq = Button(root, text="=", padx=91, pady=20, command=button_eq)
# but_clear = Button(root, text="clear", padx=79, pady=20, command=button_clear)



# but1.grid(row=3, column=0)
# but2.grid(row=3, column=1)
# but3.grid(row=3, column=2)

# but4.grid(row=2, column=0)
# but5.grid(row=2, column=1)
# but6.grid(row=2, column=2)

# but7.grid(row=1, column=0)
# but8.grid(row=1, column=1)
# but9.grid(row=1, column=2)

# but0.grid(row=4, column=0)

# but_add.grid(row=5, column=0)
# but_eq.grid(row=5, column=1, columnspan=2)
# but_clear.grid(row=4, column=1, columnspan=2)


root.mainloop()


