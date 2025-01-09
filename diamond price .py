import numpy as np
import pickle
import streamlit as st

# Load the model
with open(r'"D:/AIML/model.sav"', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Function to predict diamond price
def diamond_price_prediction(input_data):
    """
    Predict the price of a diamond using a trained Random Forest model.
    
    Parameters:
        input_data (tuple): A tuple containing the features of the diamond in the order:
                           (carat, cut, color, clarity, depth, table, x, y, z)
                           
    Returns:
        float: The predicted price of the diamond.
    """
    # Convert the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array for a single instance prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make the prediction
    prediction = loaded_model.predict(input_data_reshaped)
    
    return prediction[0]

# Main function for Streamlit app
def main():
    # Set the title
    st.title('Diamond Price Prediction')
    
    # Input fields for the user
    carat = st.text_input("Carat Weight (e.g., 0.7):")
    cut = st.text_input("Cut Quality (numerical encoding, e.g., 3):")
    color = st.text_input("Color Grade (numerical encoding, e.g., 4):")
    clarity = st.text_input("Clarity Grade (numerical encoding, e.g., 2):")
    depth = st.text_input("Depth Percentage (e.g., 61.5):")
    table = st.text_input("Table Percentage (e.g., 55):")
    x = st.text_input("Length (x-axis) in mm (e.g., 5.5):")
    y = st.text_input("Width (y-axis) in mm (e.g., 5.5):")
    z = st.text_input("Height (z-axis) in mm (e.g., 3.5):")
    
    # Prediction button
    if st.button("Predict Price"):
        try:
            # Convert inputs to a numeric tuple
            input_features = (
                float(carat),
                float(cut),
                float(color),
                float(clarity),
                float(depth),
                float(table),
                float(x),
                float(y),
                float(z)
            )

            # Make prediction
            predicted_price = diamond_price_prediction(input_features)
            st.success(f"The predicted diamond price is: ${predicted_price:.2f}")

        except ValueError:
            st.error("Please ensure all inputs are valid numbers.")

# Run the app
if __name__ == '__main__':
    main()
