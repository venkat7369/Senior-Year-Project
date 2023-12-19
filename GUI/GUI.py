import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import pandas as pd
from NN import predict
from RFR import predict_RFR
from GPR import predict_GPR
import csv
import copy
import random

# we can include our machine learning code as a separate module or within this script
# Here's a function to represent your model's prediction
# def predict(input_features):
#     # This is where your ML code would process the input features and return a prediction
#     return np.random.random()  # Dummy prediction

#i called the input csv file in the gui then it will take the calculated data and make the prediction then i will display the output 

# Create the main window
root = tk.Tk()
root.title('ML Model Prediction GUI')

# Create a frame for the File Input part
file_input_frame = tk.Frame(root)
file_input_frame.pack(pady=10)

# Add a label, entry box, and browse button to the frame
file_input_label = tk.Label(file_input_frame, text="Input CSV file:")
file_input_label.pack(side=tk.LEFT)

file_input_entry = tk.Entry(file_input_frame, width=50)
file_input_entry.pack(side=tk.LEFT, padx=5)

def browse_file():
    filename = filedialog.askopenfilename(filetypes=(("CSV files", "*.csv"),))
    file_input_entry.delete(0, tk.END)
    file_input_entry.insert(0, filename)

browse_button = tk.Button(file_input_frame, text="Browse", command=browse_file)
browse_button.pack(side=tk.LEFT)

# Add a button to load the data and run the prediction
def load_data_and_predict():
    try:
        # Here you would include the code to read the input data and preprocess it as needed
        input_file = file_input_entry.get()
        print(input_file)
        # input_data = pd.read_csv(input_file)
        ifile  = open(input_file, "rt")
        reader = csv.reader(ifile)

        # reader = csv.reader(ifile)
        csvdata=[]
        for row in reader:
                csvdata.append(row)
        ifile.close()
        # Assume we have a function `prepare_features` to process the CSV data
        # features = prepare_features(input_data)
        
        # Get predictions from your model
        # predict(csvdata)

        
        # Output predictions to a messagebox or another widget
        # messagebox.showinfo("Predictions", str(predictions))
        return csvdata
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Example function to run the chosen model (you will need to replace with your actual code)
def run_model():
    algorithm = algo_combo.get()
    csvdata = load_data_and_predict()
    if algorithm == 'Neural Network':
        # Run Neural Network
        # You will need to include your actual neural network code here
        print("Running Neural Network...")
        predict(csvdata)
    elif algorithm == 'Random Forest':
        # Run Random Forest
        # You will need to include your actual random forest code here
        print("Running Random Forest...")
        predict_RFR(csvdata)
    elif algorithm == 'Gaussian Regression':
        # Run Gaussian Regression
        # You will need to include your actual gaussian regression code here
        print("Running Gaussian Regression...")
        predict_GPR(csvdata)
    else:
        print("Please select an algorithm.")

# predict_button = tk.Button(root, text="Load Data and Predict", command=load_data_and_predict)
# predict_button.pack(pady=20)

# Label
label = tk.Label(root, text="Choose an algorithm:")
label.pack()

# Combobox for selecting the algorithm
algo_combo = ttk.Combobox(root, values=['Neural Network', 'Random Forest','Gaussian Regression'])
algo_combo.pack()

# Button to run the model
run_button = tk.Button(root, text="Run Model", command=run_model)
run_button.pack()

# Run the application
root.mainloop()

# Start the Tkinter event loop
root.mainloop()
