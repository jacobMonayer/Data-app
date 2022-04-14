import streamlit as st
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import time
import torch
import cv2
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from img_classifier import our_image_classifier
import firebase_bro

# Just making sure we are not bothered by File Encoding warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

def input_p(img):
    img = img.float()   
    img = img[None, ...]
    # img = img.permute(0,3,2,1)
    return img 
def main():
    # Metadata for the web app
    st.set_page_config(
    page_title = "Title of the webpage",
    layout = "centered",
    page_icon= ":shark:",
    initial_sidebar_state = "collapsed",
    )
    menu = ['Home', 'About', 'Contact', 'Feedback']
    choice = st.sidebar.selectbox("Menu", menu)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labels = {0:'Septoria' ,
         1:'Powdery Mildew',
         2:'Healthy' ,
         3:'Tobacco Mosiac Virus',
         4: 'Spider Mites',
         5:'Calcium Deficiency' ,
         6:'Magnesium Deficiency' }

    if choice == "Home":
        # Let's set the title of our awesome web app
        st.title('Title of your Awesome App')
        # Now setting up a header text
        st.subheader("By Your Cool Dev Name")
        # Option to upload an image file with jpg,jpeg or png extensions
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
        # When the user clicks the predict button
        if st.button("Predict"):
        # If the user uploads an image
            if uploaded_file is not None:
                # Opening our image
                image = Image.open(uploaded_file)
                # # Send our image to database for later analysis
                # firebase_bro.send_img(image)
                # Let's see what we got
                st.image(image,use_column_width=True)
                st.write("")
                try:
                    with st.spinner("The magic of our AI has started...."):
                        # model = torch.load('/gdrive/MyDrive/Final_Project/Project_Code/model_checkpoints/resnet50_epoch35')
                        # #st.error("after loading the model")
                        # #img = Image.open("/content/img_ai_app_boilerplate/model/H307_2.jpg")
                        # # open_cv_image = np.array(image) 
                        # # # Convert RGB to BGR 
                        # # img = open_cv_image[:, :, ::-1].copy() 
                        # img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        # transform = A.Compose([
                        # A.Resize(height=224, width=224),
                        # A.Normalize(),
                        # ToTensorV2(),
                        # ])
                        # # img = torch.from_numpy(img)
                        # img = transform(image=img)["image"]
                        # img = input_p(img)
            
                        # img = img.to(device)
                        # # st.error("img to device")
                        # output = model(img)
                        # # st.error("model")
                        # m = nn.Sigmoid()  
                        # # st.error("sigmoid")
                        # pred = m(output)
                        # # st.error("after labeling")
                        # pred_max = torch.argmax(pred, dim = 1)
                        # label_id = pred_max.tolist()[0]
                        # # st.error(label_id) 
                        # label = labels[label_id]

                        label_id = our_image_classifier(image)
                        label = labels[label_id]
                    st.success("We predict this image to be: " + label)
                    text = {0:'Septoria cannabis is a species of plant pathogen from the  genus Septoria that \n causes the disease commonly known as Septoria leaf spot. Early symptoms of infection  \n are concentric white lesions on the vegetative leaves of cannabis plants,\n followed by chlorosis and necrosis of the leaf until it is ultimately overcome by disease \n and all living cells are then killed. ' ,
                      1:'Powdery Mildew is arguable the most destructive Cannabis pest.\n It is an obligate biotroph that can vascularize into the plant tissue \n and remain invisible to a grower.It tends to emerge and sporulate 2 weeks into flowering \n thus destroying very mature crop with severe economic consequences.  \n It is believed to travel in clones and it’s is not known if it travels in seeds.\n Early detection and eradication may be the safest approach.',
                      2:'Your Plant is healthy' ,
                      3:'TMV is a virus that is commonly found in tobacco plants which causes splotchy or twisted leaves, \n strange mottling symptoms, slowed growth, and reduced yields. \n TMV has spread to several other species of plants besides tobacco, \n including cannabis plants. \n Plants that get infected by TMV may not grow as fast or yield as well as they could have',
                      4: 'Spider mites are part of the mite family and are related to spiders, ticks, and other mites. \n Although they’re a common cannabis pest, they can be very difficult to \n get rid of.Spider mites have tiny sharp mouths that pierce individual plant cells and suck out the contents. \n  This is what results in the tiny yellow, orange or white speckles  \n you see on your plant leaves.Spider mites are common cannabis pests, \n especially when growing in soil. Although less common in hydroponics, \n spider mites can show up in any setup where cannabis is being cultivated.',
                      5:'Calcium deficiencies can be hard to diagnose. The leaves can have dead spots,  \n crinkling, small brown spots, stunted growth, small or distorted leaves, curled tips,  \n and dark green around the affected spots. Best way to restore is to \n  flush your plants with clean pH neutral water, and then resupply the water nutrient solution.' ,
                      6:'Magnesium deficiency will give the veins and outer edges of  \n your plant a light green-yellow color. Magnesium is a mobile nutrient and \n  can travel from old leaves to new ones. Magnesium deficiency happens when  \n the pH level is too low. Often the magnesium is present,  \n but the roots cannot absorb it, therefore adding a magnesium  \n supplement won’t help your plants.  \n The best way to get that magnesium absorbed is to increase the pH.' }
                    st.text(text[label_id])
                    #rating = st.slider("Do you mind rating our service?",1,10)
                except:
                    st.error("We apologize something went wrong 🙇🏽‍♂️")
            else:
                st.error("Can you please upload an image 🙇🏽‍♂️")

    elif choice == "Contact":
        # Let's set the title of our Contact Page
        st.title('Get in touch')
        def display_team(name,path,affiliation="",email=""):
            '''
            Function to display picture,name,affiliation and name of creators
            '''
            team_img = Image.open(path)

            st.image(team_img, width=350, use_column_width=False)
            st.markdown(f"## {name}")
            st.markdown(f"#### {affiliation}")
            st.markdown(f"###### Email {email}")
            st.write("------")

        display_team("Your Awesome Name", "./assets/profile_pic.png","Your Awesome Affliation","hello@youareawesome.com")

    elif choice == "About":
        # Let's set the title of our About page
        st.title('About us')

        # A function to display the company logo
        def display_logo(path):
            company_logo = Image.open(path)
            st.image(company_logo, width=350, use_column_width=False)

        # Add the necessary info
        display_logo("./assets/profile_pic.png")
        st.markdown('## Objective')
        st.markdown("Write your company's objective here.")
        st.markdown('## More about the company.')
        st.markdown("Write more about your country here.")

    elif choice == "Feedback":
        # Let's set the feedback page complete with a form
        st.title("Feel free to share your opinions :smile:")

        first_name = st.text_input('First Name:')
        last_name = st.text_input('Last Name:')
        user_email = st.text_input('Enter Email: ')
        feedback = st.text_area('Feedback')

        # When User clicks the send feedback button
        if st.button('Send Feedback'):
            # # Let's send the data to a Database to store it
            # firebase_bro.send_feedback(first_name, last_name, user_email, feedback)

            # Share a Successful Completion Message
            st.success("Your feedback has been shared!")

if __name__ == "__main__":
    main()
