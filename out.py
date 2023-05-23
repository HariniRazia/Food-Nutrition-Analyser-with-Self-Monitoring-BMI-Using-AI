
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
import os
import numpy as np
from tkinter import messagebox
import random

model = load_model('foodout_model1.h5')

train_data_dir = './dataset/train'
test_data = []
test_labels = []
y_pred_nm = []
img_width, img_height = 224, 224
batch_size = 18
image_size = (224, 224)

train_labels = ['Apple','Banana','Biriyani','Burger','Cake','Chappathi','Chips','Corn','Dosa','French Fries','Greens',
'Gulab Jamun','Ice Cream','Idly','Jileabi','Mango','Medu Vadai','Mysore Pak','Noodles',
'Orange','Paniyaram','Paneer Tikka','Parotta','Pine Apple','Pizza','Poori','Sambar','Samosa','Shawarma','Soft Drinks',
'Tandoori Chicken','Toasted Bread','Upma','Ven Pongal','White Rice']
cal=["Cal: 95Kcal Fat - 0g Protein - 1g Crabs - 25g \nBenefits - Rich in fiber and antioxidants", #Apple
     "Cal: 110Kcal Fat - 0g Protein - 1g Crabs - 28g \nBenefits - Rich in Vitamin C & B6", #Banana
     "Cal: 720Kcal Fat - 20g Protein - 16g Crabs - 31g \nBenefits - Complete package of proteins \nEffects -  Weight Gain", #Biriyani
     "Cal: 266Kcal Fat - 11g Protein - 13g Crabs - 31g \nEffects -  Unhealthy & Weight Gain", #Burger
     "Cal: 257Kcal Fat - 12g Sugar - 28g Crabs - 38g \nEffects -  Unhealthy, Weight Gain & Causes Acne", #Cake
     "Cal: 70Kcal Fat - 1g Protein - 3g Crabs - 15g \nBenefits - Helps to stay fit \nEffects -  Increases your cravings for more food", #Chapathi
     "Cal: 160Kcal Fat - 10g Protein - 2g Crabs - 15g \nEffects -  Oil fried foods has higher risk of heart problems", #Chips
     "Cal: 86Kcal Fat - 2g Protein - 4g Crabs - 19g \nBenefits - Rich in fiber and Vitamin C ", #Corn
     "Cal: 82Kcal Fat - 4g Protein - 3g Crabs - 17.8g \nBenefits - Rich in Vitamin A and C", #Dosa
	 "Cal: 365Kcal Fat - 19g Protein - 1.96g Crabs - 18.5g \nBenefits - Potatoes are rich in potassium \nEffects - Oil fried foods has higher risk of heart problems", #FrenchFries
	 "Cal: 63Kcal Fat - 2g Protein - 6g Crabs - 11g \nBenefits - Rich in Vitamin A,K,E,C and E", #Greens
	 "Cal: 160Kcal Fat - 7g Protein - 4g Crabs - 24g \nBenefits - Source of Vitamin A and Calcium  \nEffects -  Risk of Heart Problems", #GJ
	 "Cal: 207Kcal Fat - 21g Protein - 4g Crabs - 24g \nBenefits - Milk IceCream is a source of vitamin A and D \nEffects -  Weight Gain", #Ice cream
	 "Cal: 39Kcal Fibre - 5g Protein - 5g Crabs - 10g \nBenefits - Excellent Heart-Healthy Nutrition ", #Idly
	 "Cal: 150Kcal Fat - 4g Protein - 2g Crabs - 29g \nEffects -  Weight Gain", #Jileabi
	 "Cal: 99Kcal Fat - 0g Protein - 2g Crabs - 24g \nBenefits - Rich in Vitamin C,B6,A,E and K", #Mango
	 "Cal: 97Kcal Fat - 6g Protein - 4g Crabs - 9g \nBenefits - Source of Vitamin A and B9 \nEffects -  Oil fried foods has higher risk of heart problems", #Vadai
	 "Cal: 564Kcal Fat - 22g Protein - 3g Crabs - 68g \nBenefits - Source of protein and fiber  \nEffects -  Weight Gain", #MysorePak
	 "Cal: 198Kcal Fat - 11g Protein - 4g Crabs - 27g \nBenefits - Source of phosphorus, Magnesium and Fiber  \nEffects -  High Sodium & Weight Gain", #Noodles
	 "Cal: 66Kcal Fat - 0g Protein - 1g Crabs - 15g \nBenefits - Rich in Vitamin C and Protects cell damage ", #Orange
	 "Cal: 107Kcal Fat - 3g Protein - 2g Crabs - 16g \nBenefits - Source of Carbs,manganese and potassium \nEffects -  Weight Gain", #Paniyaram
	 "Cal: 278Kcal Fat - 198g Protein - 44g Crabs - 31g \nBenefits - Source pf Healthy fats and Protein \nEffects -  Oil fried foods has higher risk of heart problems", #Paneertikka
	 "Cal: 482Kcal Fat - 255g Protein - 29g Crabs - 198g \nBenefits - Source of fibre  \nEffects -  indigestion & Weight Gain", #Parotta
	 "Cal: 50Kcal Fat - 0g Potassium - 109mg Crabs - 13g \nBenefits - Rich in antioxidants ", #Pineapple
	 "Cal: 372Kcal Fat - 14g Protein - 13g Crabs - 37g \nEffects -  Weight Gain", #Pizza
	 "Cal: 101Kcal Fat - 67g Protein - 5g Crabs - 30g \nBenefits - Source of phosphorus \nEffects - Oil fried foods has higher risk of heart problems", #Poori
	 "Cal: 139Kcal Fat - 10g Protein - 5g Crabs - 19g \nBenefits - Rich in Protein and antioxidants", #Sambar
	 "Cal: 91Kcal Fat - 71g Protein - 6g Crabs - 32g \nEffects -  Oil fried foods has higher risk of heart problems", #Samosa
	 "Cal: 512Kcal Fat - 34g Protein - 6g Crabs - 77g \nBenefits - Source of magnesium,fiber and sodium \nEffects -  Weight Gain", #Shawarma
	 "Cal: 149Kcal Fat - 0g Sugar - 41g Sodium - 30mg \nBenefits - Helps in digestion  \nEffects - Teeth and kidney problems", #Drinks
	 "Cal: 263Kcal Fat - 12g Protein - 31g Crabs - 7g \nBenefits - Good source of Protein  \nEffects - High Cholesterol and Weight Gain", #Tandoori Chicken
	 "Cal: 89Kcal Fat - 2g Protein - 4g Crabs - 16g \nBenefits - Lower glycemic index \nEffects - Indigestion & Weight Gain", #Bread
	 "Cal: 150Kcal Fat - 5g Protein - 4g Crabs - 24g \nBenefits - Source of fiber, vitamins and healthy fats", #Upma
	 "Cal: 212Kcal Fat - 74g Protein - 22g Crabs - 116g \nBenefits - Maintains Protein-Carb Balance ", #Ven Pongal
	 "Cal: 242Kcal Fat - 2g Protein - 5g Crabs - 54g \nBenefits - Source of Vitamins and Minerals \nEffects -  Pollished white rice are unhealthy"  #Rice
]
over=["Eat a balanced, calorie-controlled diet as recommended by your health professional",
"Join a local weight loss group",
"Take up activities such as fast walking, jogging, swimming or tennis for 150 to 300 minutes a week.",
"Eat slowly and avoid situations where you know you could be tempted to overeat.",
"Drinking lots of water will not only help with your digestion but it will also boost your metabolism, cleanse your body of waste, and reduce bloating.",
"Adding protein to your diet is one of the quickest ways to get rid of overweight without additional exercise.",
"Green tea is a rich source of catechins and caffeine that speed up your metabolism and aid in burning fat.",
"Coconut oil contains medium-chain triglycerides (MCTs) which makes it an effective weight loss remedy.",
"Drinking Apple cider vinegar every day can boost your health and also obtain weight loss."
"The mixture of lemon juice and honey to a cup of warm water is a popular remedy for weight loss.",
"Garlic, a natural herb is responsible for boosting energy levels, burn calories and keep you fit.",
"Cinnamon has its own ways to help you combat obesity and reduce weight."
]

under=["Eggs are the healthiest and best option to add to your diet plan for weight gain.",
"Chicken being rich in protein helps gain weight or kilos. It is also packed with nutrients that are essential for the body.",
"Potatoes have healthy starch and important nutrients and fiber that further boosts your calorie intake, and increase muscle glycogen stores.",
"Paneer contains a fair amount of healthy fat and proteins.",
"Avoid weight gaining with junk foods and alcohol.",
"Take 0.8g per/kg protein per kilogram of your body weight",
"Milk is rich in nutrition profile including calcium, protein, vitamins, minerals, etc.",
"Fishes such as salmon are rich in fatty acids like omega-3 fatty acids. Consuming them can make you gain weight easily and effectively.",
"Whole wheat bread has both fiber and carbs that makes it good for weight.",
"Eating some beans, pulses, fish, eggs, meat and other protein.",
"Increase your calorie intake by eating foods like milky puddings and cheesy main courses, or vegetarian or vegan alternatives.",
"Replace 1 cup of tea or coffee each day with a cup of warm full-fat milk or a dairy-free alternative such as soya milk."
]
           
def get_height():
    height = float(ENTRY2.get())
    return height


def get_weight():
    weight = float(ENTRY1.get())
    return weight


def calculate_bmi(a=""):   # "a" is there because the bind function gives an argument to the function....
    print(a)
    d=random.randint(0,9)
    try:
        height = get_height()
        weight = get_weight()
        height = height / 100.0
        bmi = weight / (height ** 2)
    except ZeroDivisionError:
        messagebox.showinfo("Result", "Please enter positive height!!")
    except ValueError:
        messagebox.showinfo("Result", "Please enter valid data!")
    else:
        if bmi <= 15.0:
            res = "Your BMI is " + str(bmi) + "\nCondition: Very severely underweight!!"
            messagebox.showinfo("Result", res)
            messagebox.showinfo("SUGGESTION", under[d])
        elif 15.0 < bmi <= 16.0:
            res = "Your BMI is " + str(bmi) + "\nCondition: Severely underweight!"
            messagebox.showinfo("Result", res)
            messagebox.showinfo("SUGGESTION", under[d])
        elif 16.0 < bmi < 18.5:
            res = "Your BMI is " + str(bmi) + "\nCondition: Underweight!"
            messagebox.showinfo("Result", res)
            messagebox.showinfo("SUGGESTION", under[d])
        elif 18.5 <= bmi <= 25.0:
            res = "Your BMI is " + str(bmi) + "\nCondition: Normal."
            messagebox.showinfo("Result", res)
        elif 25.0 < bmi <= 30:
            g=over[d]
            res = "Your BMI is " + str(bmi) + "\nCondition: Overweight."
            messagebox.showinfo("Result", res)
            messagebox.showinfo("SUGGESTION", over[d])
        elif 30.0 < bmi <= 35.0:
            g=over[d]
            res = "Your BMI is " + str(bmi) + "\nCondition: Moderately obese!"
            messagebox.showinfo("Result", res)
            messagebox.showinfo("SUGGESTION", over[d])

            g=over[d]
        elif 35.0 < bmi <= 40.0:
            res = "Your BMI is " + str(bmi) + "\nCondition: Severely obese!"
            g=over[d]
            messagebox.showinfo("Result", res)
            messagebox.showinfo("SUGGESTION", over[d])

        else:
            res = "Your BMI is " + str(bmi) + "\nCondition: Super obese!!"
            g=over[d]
            messagebox.showinfo("Result", res)
            messagebox.showinfo("SUGGESTION", over[d])
 
def model_predict(img_path):
        print ("image path",img_path)
        img = image.load_img(img_path, target_size=image_size)
        x 	= image.img_to_array(img)
        x 	= np.expand_dims(x, axis=0)
        x /= 255. 
        test_data.append(x)
        Y_pred = model.predict(x)
        y_pred = np.argmax(Y_pred, axis=1)
        y_pred_nm.append(y_pred[0])
        print(y_pred[0])

        print('Prediction:', train_labels[y_pred[0]])
        result=train_labels[y_pred[0]]
        pesti=cal[y_pred[0]]
        return result,pesti
        

def select_image():
	global panelA, panelB

	path = filedialog.askopenfilename()
    	# ensure a file path was selected
	if len(path) > 0:
		# load the image from disk, convert it to grayscale, and detect
		# edges in it
		image = cv2.imread(path)
		image=cv2.resize(image,(500,400)) 
		c,pesti=model_predict(path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(image)
		image = ImageTk.PhotoImage(image)

        
        		# if the panels are None, initialize them
		if panelA is None :
			# the first panel will store our original image
			panelA = Label(image=image)
			panelA.image = image
			panelA.place(x=175, y=98)



		else:
			panelA.configure(image=image)
			
			panelA.image = image
			
            
		lbl.configure(text=pesti)

		Btn1.configure(text="DETECTED: "+c)

root = Tk()
root.title('A NUTRITION ANALYZER DRIVEN BY AI FOR HEALTH CONSCIOUS PEOPLE')
root.state('zoomed')
root['bg']="#DB96A9"
lbl2 = Label(root, text = "FOOD CLASSIFICATION",width=20,font=("Arial", 17),bg="#5ff255")
lbl2.place(x=285, y=40)
panelA = None
outlabel=None
btn = Button(root,width=36, text="SELECT AN IMAGE",font=("Arial", 15), command=select_image,bg="#5ff255")
btn.place(x=220, y=540)
Btn1 = Label(root, text="",font=("Arial", 20), bg="#DB96A9", fg='black')
Btn1.place(x=200, y=610)
lbl = Label(root, text="",font=("Arial", 15), wraplength=500,compound=LEFT, justify=LEFT,bg="#DB96A9", fg='black')
lbl.place(x=200, y=660) 

LABLE = Label(root, bg="#5ff255", text="BMI CALCULATOR", font=("Arial", 15))
LABLE.place(x=1060, y=100)
LABLE1 = Label(root, bg="#cef0f1", text="Enter Weight (in kg):", bd=6,font=("Arial", 10), pady=5)
LABLE1.place(x=1000, y=200)
ENTRY1 = Entry(root,  width=26, font="Roboto 15")
ENTRY1.place(x=1000, y=260)
LABLE2 = Label(root, bg="#cef0f1", text="Enter Height (in cm):", bd=6,font=("Arial", 10), pady=5)
LABLE2.place(x=1000, y=340)
ENTRY2 = Entry(root, width=26, font="Roboto 15")
ENTRY2.place(x=1000, y=400)
BUTTON = Button(bg="#5ff255",text="CALCULATE",bd = 5,width=14, padx=4, pady=4, command=calculate_bmi, font=("Arial", 12))
BUTTON.place(x=1065, y=500)

root.mainloop()