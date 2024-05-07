import streamlit as st
from tensorflow import keras
import cv2
import numpy as np
import os
from streamlit_option_menu import option_menu
import json
from streamlit_extras.stylable_container import stylable_container
from streamlit_space import space
from streamlit_lottie import st_lottie_spinner


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


# Page Loader
st.set_page_config(
    page_title="DermaCare",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "The Skin Disease Recognition web app leverages a trained model to predict skin diseases from "
                 "uploaded images. Users can simply upload skin lesion images for instant analysis and identification "
                 "of the skin condition."
    }
)

# Navigation Bar
with st.sidebar:
    selected = option_menu("Dashboard", ["Home", 'Prediction', 'Contact'],
                           icons=['house', 'search', 'phone'], menu_icon="cast", default_index=0)

# Home
if selected == "Home":
    title_container = st.container()
    with title_container:
        c1, _, c2 = st.columns([0.1, 0.01, 0.8])
        with c1:
            st.image("logo.png")
        with c2:
            st.title("DermaCare")
            st.markdown(
                "<p style='font-size: 20px;'>The Skin Disease Recognition web app leverages a trained model to predict skin diseases from "
                "uploaded images. Users can simply upload skin lesion images for instant analysis and identification "
                "of the skin condition.</p>", unsafe_allow_html=True)
    st.divider()

    space()
    st.markdown("<p style='font-size: 20px;'> DermaCare is a skin disease recognition web app. It details the model's "
                "implementation, including the neural network architecture and data augmentation techniques. Each "
                "skin condition, such as vitiligo, candidiasis, melanoma, and eczema, is described, highlighting "
                "symptoms, treatment, and prevention measures for user education and awareness. </p>",
                unsafe_allow_html=True)
    st.button("Get Started")
    st.divider()

    home_descrip_container1 = st.container()
    with home_descrip_container1:
        space()
        space()
        c1, _, c2 = st.columns([1, 0.2, 1])
        with c1:
            st.header("VITILIGO")
            st.markdown(
                "<p style='font-size: 20px; text-align: justify;'> ⭕Vitiligo is a skin condition characterized by the loss of pigmentation, leading to white "
                "patches on the skin. It results from the immune system attacking melanocytes,"
                "causing depigmentation. <br><br> ⭕Vitiligo can affect any part of the body and may have a significant "
                "impact on a person's self-esteem. <br><br> ⭕Treatment options include topical corticosteroids, "
                "light therapy, and skin grafting. </p>", unsafe_allow_html=True)
        with c2:
            st.image("vitiligo.webp", width=500)
        space()
        space()
    st.divider()

    home_descrip_container2 = st.container()
    with home_descrip_container2:
        space()
        space()
        c1, _, c2 = st.columns([1, 0.2, 1])
        with c1:
            st.header("CANDIDIASIS")
            st.markdown(
                "<p style='font-size: 20px; text-align: justify;'> ⭕Candidiasis, a fungal infection caused by Candida "
                "species, includes thrush, genital candidiasis, and invasive forms. <br><br> ⭕Thrush shows as white patches, "
                "genital candidiasis causes itching and redness, while invasive candidiasis impacts deeper tissues. "
                "It commonly affects those with weakened immunity. <br><br> ⭕Treatment involves antifungal medications, "
                "oral or topical. Prevention focuses on hygiene and avoiding triggers. Quick medical attention is "
                "essential to prevent complications. </p>", unsafe_allow_html=True)

        with c2:
            st.image("candidiasis.webp", width=450)
        space()
        space()
    st.divider()

    home_descrip_container3 = st.container()
    with home_descrip_container3:
        space()
        space()
        c1, _, c2 = st.columns([1, 0.2, 1])
        with c1:
            st.header("MELANOMA")
            st.markdown(
                "<p style='font-size: 20px; text-align: justify;'> ⭕Melanoma is a form of skin cancer stemming from "
                "melanocytes—cells producing pigment. It primarily surfaces as new moles or changes in existing ones. "
                "Melanomas can also occur within the mouth, intestines, or eyes. <br><br> ⭕Early detection and intervention are "
                "vital, as melanoma can spread rapidly to other parts of the body. Risk factors include excessive UV "
                "exposure and a family history of the condition.  <br><br>⭕Regular "
                "skin checks and sun protection are crucial preventive measures. Seeking medical advice promptly upon "
                "observing any irregularities in moles or the skin is imperative for the most favorable "
                "outcomes.</p>", unsafe_allow_html=True)
        with c2:
            st.markdown("<p style='margin-top: 80px;'> </p>", unsafe_allow_html=True)
            st.image("melnoma.jpg", width=500)
        space()
        space()
    st.divider()

    home_descrip_container4 = st.container()
    with home_descrip_container4:
        space()
        space()
        c1, _, c2 = st.columns([1, 0.2, 1])
        with c1:
            st.header("ECZEMA")
            st.markdown(
                "<p style='font-size: 20px; text-align: justify;'> ⭕Eczema, or atopic dermatitis, is a chronic skin "
                "condition characterized by redness, itching, and inflammation. It commonly affects children but can "
                "persist into adulthood. <br><br>⭕Eczema flare-ups are often triggered by irritants, allergens, stress, "
                "or changes in weather. Proper skincare, moisturizing, and avoiding known triggers are key to "
                "managing eczema. <br><br>⭕Seeking guidance from a dermatologist or healthcare "
                "provider is essential for personalized treatment and long-term management of eczema. </p>",
                unsafe_allow_html=True)

        with c2:
            st.markdown("<p style='margin-top: 80px;'> </p>", unsafe_allow_html=True)
            st.image("eczema.webp", width=500)
        space()
        space()
    st.divider()


# Prediction
elif selected == "Prediction":
    # Load the trained model
    model = keras.models.load_model('skin_disease_classification_model.h5')

    # Class names
    path = 'train'
    class_names = sorted(os.listdir(path))


    def predict_disease(image):
        # Resize the input image to match the expected input shape of the model
        img = cv2.resize(image, (64, 64))  # Resize to (64, 64)
        img = np.reshape(img, (1, 64, 64, 3))  # Reshape the image to match the input shape
        img = img / 255.0  # Normalize the image

        # Make prediction
        prediction = model.predict(img)[0]
        max_index = np.argmax(prediction)
        predicted_class = class_names[max_index]

        return predicted_class, prediction


    # Streamlit app
    st.title('Skin Disease Recognition')

    uploaded_file = st.file_uploader("Upload an image of skin lesion", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', width=200)

        if st.button('Predict'):
            predicted_class, prediction = predict_disease(image)
            st.success(f"The skin disease in the image is predicted as: {predicted_class}")

            # Recommendation System

            if predicted_class == "vitiligo":
                with st.expander("SYSTEM OF VITILIGO RECOMMENDATIONS"):
                    vitiligo_container1 = st.container()
                    with vitiligo_container1:
                        c1,_, c2 = st.columns([1.3,0.05,  0.7])
                        with c1:
                            st.header("SKIN CARE TREATMENT")
                            st.markdown(
                                "<p style='font-size: 20px; text-align: justify;'> ⭕Skin cancer treatment for "
                                "vitiligo involves using various methods to manage the condition. Common treatments "
                                "include topical corticosteroids to reduce inflammation and encourage repigmentation "
                                "of the skin. <br><br> ⭕Phototherapy, such as UVB therapy, can also help stimulate melanocytes "
                                "to produce pigment in affected areas. In more severe cases, surgical options like "
                                "skin grafting or melanocyte transplantation may be considered to restore "
                                "pigmentation. <br><br> ⭕It is essential for individuals with vitiligo to work closely with "
                                "dermatologists to determine the most suitable treatment plan based on the extent and "
                                "progression of their condition, aiming for both cosmetic improvement and overall "
                                "skin health.</p>", unsafe_allow_html=True)
                        with c2:
                            st.markdown("<p style='margin-top: 40px;'> </p>", unsafe_allow_html=True)
                            st.image("heartbeat.png",
                                     width=400)
                        space()
                        space()
                    st.divider()
                    vitiligo_container2 = st.container()
                    with vitiligo_container2:
                        c1, _, c2 = st.columns([1.3, 0.05, 0.7])
                        with c1:
                            st.header("DO'S AND DONT'S")
                            st.markdown(
                                "<p style='font-size: 20px; text-align: justify;'> When managing vitiligo, there are "
                                "important do's and don'ts to consider. <br><br> ⭕Do protect your skin from sun exposure by "
                                "using sunscreen and wearing protective clothing to prevent sunburn in depigmented "
                                "areas. <br><br> ⭕Do consult a dermatologist for personalized treatment options, "
                                "including topical corticosteroids, phototherapy, or surgical interventions. <br><br> ⭕Do seek "
                                "support from vitiligo support groups and counseling to address any emotional impact. "
                                "Don't use harsh chemicals or treatments that may aggravate the skin. <br><br> ⭕Don't neglect "
                                "regular skin checks for signs of skin cancer, especially in depigmented areas. <br><br> ⭕"
                                "Finally, don't underestimate the importance of self-care and self-acceptance in "
                                "coping with vitiligo.</p>", unsafe_allow_html=True)
                        with c2:
                            st.markdown("<p style='margin-top: 120px;'> </p>", unsafe_allow_html=True)
                            st.image("allergy.png",
                                     width=400)
                        space()
                        space()

                        st.divider()
                        vitiligo_container2 = st.container()
                        with vitiligo_container2:
                            c1, _, c2 = st.columns([1.3, 0.05, 0.7])
                            with c1:
                                st.header("DIET PLAN")
                                st.markdown(
                                    "<p style='font-size: 20px; text-align: justify;'> ⭕A well-balanced diet plan for "
                                    "vitiligo focuses on supporting overall skin health and potential repigmentation. "
                                    "Include foods rich in antioxidants like fruits, vegetables, and green tea to "
                                    "help combat oxidative stress. <br><br> ⭕Incorporate foods high in vitamins C, E, and D, "
                                    "as well as minerals like copper and zinc known for their role in skin health. "
                                    "Consider adding foods with phenylalanine content like dairy, meat, "
                                    "and soy products, as this amino acid may support repigmentation. <br><br> ⭕Avoiding "
                                    "trigger foods that may worsen autoimmune responses is also crucial. Consulting a "
                                    "nutritionist or dermatologist for a personalized diet plan tailored to "
                                    "individual needs is highly recommended.</p>", unsafe_allow_html=True)
                            with c2:
                                st.markdown("<p style='margin-top: 80px;'> </p>", unsafe_allow_html=True)
                                st.image("diet.png",
                                         width=400)
                            space()
                            space()

            elif predicted_class == "Tinea Ringworm Candidiasis and other Fungal Infections":
                    with st.expander("SYSTEM OF CANDIDIASIS RECOMMENDATIONS"):
                        vitiligo_container2 = st.container()
                        with vitiligo_container2:
                            c1, _, c2 = st.columns([1.3, 0.05, 0.7])
                            with c1:
                                st.header("SKIN CARE TREATMENT")
                                st.markdown(
                                    "<p style='font-size: 20px; text-align: justify;'> ⭕When addressing skin cancer "
                                    "treatment for candidiasis, it's essential to understand that candidiasis is a "
                                    "fungal infection rather than a form of skin cancer. Treatment for candidiasis "
                                    "typically involves antifungal medications to eliminate the fungal overgrowth on "
                                    "the skin. <br><br> ⭕Topical antifungal creams, ointments, or oral medications may be "
                                    "prescribed based on the severity and location of the infection. Proper hygiene "
                                    "practices, keeping the affected area clean and dry, and avoiding factors that "
                                    "promote fungal growth are also important aspects of managing candidiasis. "
                                    "<br><br> ⭕Consulting a healthcare provider for an accurate diagnosis and appropriate "
                                    "treatment plan tailored to the individual's needs is crucial for effectively "
                                    "addressing candidiasis.</p>",
                                    unsafe_allow_html=True)
                            with c2:
                                st.markdown("<p style='margin-top: 40px;'> </p>", unsafe_allow_html=True)
                                st.image("test.png",
                                         width=400)
                            space()
                            space()
                        st.divider()
                        vitiligo_container2 = st.container()
                        with vitiligo_container2:
                            c1, _, c2 = st.columns([1.3, 0.05, 0.7])
                            with c1:
                                st.header("DO'S AND DONT'S")
                                st.markdown(
                                    "<p style='font-size: 20px; text-align: justify;'> When dealing with candidiasis, "
                                    "specific do's and don'ts are vital for effective management. <br><br> ⭕Do practice good "
                                    "hygiene by keeping the affected areas clean and dry to prevent further fungal "
                                    "growth. <br><br> ⭕Do wear loose-fitting, breathable clothing to promote airflow and reduce "
                                    "moisture, creating an environment less favorable for candida overgrowth. <br><br> ⭕Do use "
                                    "antifungal medications as prescribed by a healthcare professional to target the "
                                    "infection directly. <br><br> ⭕Don't use harsh soaps or irritating products that can "
                                    "disrupt the natural balance of the skin and exacerbate candidiasis. <br><br> ⭕Don't "
                                    "scratch or aggravate the infected areas to prevent spreading the infection or "
                                    "causing skin damage. <br><br> ⭕Adhering to these do's and don'ts can help in effectively "
                                    "managing candidiasis and promoting skin health.</p>", unsafe_allow_html=True)
                            with c2:
                                st.markdown("<p style='margin-top: 120px;'> </p>", unsafe_allow_html=True)
                                st.image("choice.png",
                                         width=400)
                            space()
                            space()

                            st.divider()
                            vitiligo_container2 = st.container()
                            with vitiligo_container2:
                                c1, _, c2 = st.columns([1.3, 0.05, 0.7])
                                with c1:
                                    st.header("DIET PLAN")
                                    st.markdown(
                                        "<p style='font-size: 20px; text-align: justify;'> ⭕When crafting a diet plan "
                                        "for candidiasis, focus on reducing foods that promote yeast growth. "
                                        "Emphasize a diet rich in non-starchy vegetables, low-sugar fruits, "
                                        "lean protein sources, and healthy fats. Include probiotic-rich foods like "
                                        "yogurt and kefir to support gut health and balance the microbiome. "
                                        "<br><br> ⭕Incorporate anti-fungal foods such as garlic, coconut oil, and apple cider "
                                        "vinegar known for their candida-fighting properties. Limit sugary foods, "
                                        "refined carbohydrates, and alcohol, as these can exacerbate yeast "
                                        "overgrowth. <br><br> ⭕Hydration is essential to flush out toxins, so prioritize water "
                                        "intake. Consulting a healthcare provider or a nutritionist for personalized "
                                        "dietary recommendations can aid in managing candidiasis effectively.</p>",
                                        unsafe_allow_html=True)
                                with c2:
                                    st.markdown("<p style='margin-top: 80px;'> </p>", unsafe_allow_html=True)
                                    st.image("checklist.png",
                                        width=400)
                                space()
                                space()
            elif predicted_class == "Melanoma Skin Cancer Nevi and Moles":
                with st.expander("SYSTEM OF MELANOMA RECOMMENDATIONS"):
                    vitiligo_container2 = st.container()
                    with vitiligo_container2:
                        c1, _, c2 = st.columns([1.3, 0.05, 0.7])
                        with c1:
                            st.header("SKIN CARE TREATMENT")
                            st.markdown(
                                "<p style='font-size: 20px; text-align: justify;'> ⭕Treatment for melanoma skin "
                                "cancer typically involves surgical removal of the tumor, along with some surrounding "
                                "healthy tissue to ensure complete excision. <br><br> ⭕Depending on the stage and spread of the "
                                "cancer, additional therapies such as immunotherapy, targeted therapy, chemotherapy, "
                                "or radiation therapy may be recommended to target any remaining cancer cells and "
                                "reduce the risk of recurrence. Immunotherapy, which harnesses the body's immune "
                                "system to fight cancer cells, has shown promising results in treating advanced "
                                "melanoma. <br><br> ⭕Additionally, ongoing monitoring and regular skin checks are essential to "
                                "detect any potential recurrence or new skin cancers, emphasizing the significance of "
                                "early detection and timely intervention in managing melanoma.</p>",
                                unsafe_allow_html=True)
                        with c2:
                            st.markdown("<p style='margin-top: 40px;'> </p>", unsafe_allow_html=True)
                            st.image("hydrated.png",
                                     width=400)
                        space()
                        space()
                    st.divider()
                    vitiligo_container2 = st.container()
                    with vitiligo_container2:
                        c1, _, c2 = st.columns([1.3, 0.05, 0.7])
                        with c1:
                            st.header("DO'S AND DONT'S")
                            st.markdown(
                                "<p style='font-size: 20px; text-align: justify;'> Certainly, when it comes to "
                                "managing melanoma skin cancer, adhering to specific do's and don'ts is crucial. <br><br> ⭕Do "
                                "conduct regular skin self-exams and promptly report any changes in moles, skin, "
                                "or overall health to a healthcare professional. <br><br> ⭕Do protect the skin from excessive "
                                "UV exposure by using sunscreen, wearing protective clothing, and seeking shade, "
                                "especially during peak sun hours. <br><br> ⭕Do follow the recommended follow-up care and "
                                "surveillance schedule post-treatment to monitor for any signs of recurrence. Don't "
                                "ignore any unusual changes on the skin, such as new moles, changes in existing "
                                "moles, or unusual skin growths. <br><br> ⭕Don't disregard the importance of professional "
                                "medical advice, timely screenings, and ongoing vigilance in managing melanoma skin "
                                "cancer.</p>", unsafe_allow_html=True)
                        with c2:
                            st.markdown("<p style='margin-top: 120px;'> </p>", unsafe_allow_html=True)
                            st.image("decision.png",
                                     width=400)
                        space()
                        space()

                        st.divider()
                        vitiligo_container2 = st.container()
                        with vitiligo_container2:
                            c1, _, c2 = st.columns([1.3, 0.05, 0.7])
                            with c1:
                                st.header("DIET PLAN")
                                st.markdown(
                                    "<p style='font-size: 20px; text-align: justify;'> ⭕A well-considered diet plan "
                                    "for individuals with melanoma skin cancer aims to support overall health and "
                                    "well-being. Emphasizing a diet rich in fruits, vegetables, and whole grains can "
                                    "provide essential vitamins, minerals, and antioxidants that support the body's "
                                    "immune system and overall health. <br><br> ⭕Including sources of omega-3 fatty acids, "
                                    "such as fatty fish, flaxseeds, and walnuts, may offer anti-inflammatory "
                                    "benefits. Conversely, avoiding excessive intake of processed and red meats, "
                                    "as well as sugary and high-fat foods, is advisable. <br><br> ⭕Consulting a registered "
                                    "dietitian for personalized nutritional guidance, considering potential "
                                    "interactions with treatment, and addressing individual dietary needs is crucial "
                                    "in formulating an effective and nourishing diet plan.</p>", unsafe_allow_html=True)
                            with c2:
                                st.markdown("<p style='margin-top: 80px;'> </p>", unsafe_allow_html=True)
                                st.image("calories.png",
                                         width=400)
                            space()
                            space()
            elif predicted_class == "Eczema Photos":
                    with st.expander("SYSTEM OF ECZEMA RECOMMENDATIONS"):
                        vitiligo_container2 = st.container()
                        with vitiligo_container2:
                            c1, _, c2 = st.columns([1.3, 0.05, 0.7])
                            with c1:
                                st.header("SKIN CARE TREATMENT")
                                st.markdown(
                                    "<p style='font-size: 20px; text-align: justify;'> ⭕Skin cancer treatment for "
                                    "eczema involves a tailored approach due to the delicate nature of eczematous "
                                    "skin. Topical treatments like corticosteroids or calcineurin inhibitors may be "
                                    "used cautiously to manage eczema while addressing any potential cancerous "
                                    "lesions. <br><br> ⭕Regular skin checks by a dermatologist are essential to detect any "
                                    "suspicious changes on the skin early. In cases where skin cancer is detected, "
                                    "treatments such as surgical excision, photodynamic therapy, or immune-based "
                                    "therapies might be considered. <br><br> ⭕Close collaboration between dermatologists and "
                                    "oncologists is crucial to develop a comprehensive treatment plan that addresses "
                                    "both eczema management and skin cancer treatment effectively.</p>",
                                    unsafe_allow_html=True)
                            with c2:
                                st.markdown("<p style='margin-top: 40px;'> </p>", unsafe_allow_html=True)
                                st.image("dry_skin.png",
                                         width=400)
                            space()
                            space()
                        st.divider()
                        vitiligo_container2 = st.container()
                        with vitiligo_container2:
                            c1, _, c2 = st.columns([1.3, 0.05, 0.7])
                            with c1:
                                st.header("DO'S AND DONT'S")
                                st.markdown(
                                    "<p style='font-size: 20px; text-align: justify;'> When managing eczema, "
                                    "it's crucial to adhere to specific do's and don'ts. <br><br> ⭕Do moisturize the skin "
                                    "regularly with a gentle, fragrance-free moisturizer to maintain skin hydration "
                                    "and reduce flare-ups. <br><br> ⭕Do identify and avoid triggers such as certain soaps, "
                                    "detergents, and environmental allergens that exacerbate eczema symptoms. <br><br> ⭕Do use "
                                    "mild, non-irritating skincare products and laundry detergents to minimize skin "
                                    "irritation. <br><br> ⭕Don't scratch or rub the affected areas to prevent further skin "
                                    "damage and infection. <br><br> ⭕Don't overlook the importance of seeking professional "
                                    "medical advice for personalized treatment and management strategies to "
                                    "effectively control eczema symptoms and improve overall skin health.</p>",
                                    unsafe_allow_html=True)
                            with c2:
                                st.markdown("<p style='margin-top: 120px;'> </p>", unsafe_allow_html=True)
                                st.image("decision (1).png",
                                         width=400)
                            space()
                            space()

                            st.divider()
                            vitiligo_container2 = st.container()
                            with vitiligo_container2:
                                c1, _, c2 = st.columns([1.3, 0.05, 0.7])
                                with c1:
                                    st.header("DIET PLAN")
                                    st.markdown(
                                        "<p style='font-size: 20px; text-align: justify;'> ⭕A well-thought-out diet "
                                        "plan for eczema focuses on maintaining skin health and reducing "
                                        "inflammation. Include foods rich in omega-3 fatty acids like fatty fish, "
                                        "flaxseeds, and walnuts, known for their anti-inflammatory properties that "
                                        "may help alleviate eczema symptoms. <br><br> ⭕Incorporate fruits and vegetables high "
                                        "in antioxidants, such as berries and leafy greens, to support skin "
                                        "regeneration. Avoid potential trigger foods like dairy, gluten, "
                                        "and processed foods that may exacerbate eczema flare-ups. <br><br> ⭕Hydration is key, "
                                        "so consuming an adequate amount of water is essential to keep the skin "
                                        "hydrated from within. Consulting a dietitian for personalized guidance can "
                                        "be beneficial in creating an effective dietary plan tailored to individual "
                                        "needs.</p>",
                                        unsafe_allow_html=True)
                                with c2:
                                    st.markdown("<p style='margin-top: 80px;'> </p>", unsafe_allow_html=True)
                                    st.image("calories.png",
                                             width=400)
                                space()
                                space()
            else:
                st.error("Invalid Option")


# Contact
elif selected == 'Contact':
    st.title("☎️ Contact Details")
    st.divider()

    st.markdown("<p style='font-size: 22px;'> 📫Email: dermacareinfo@gmail.com <br><br>"
                "📱Phone: +916335555111 <br><br>"
                "📣Address: 123 Mt street, Vellore, Tamil Nadu, India - 632001 </p>", unsafe_allow_html=True)

    st.divider()
    info_container = st.container()
    with info_container:
        c1, _, c2, _, c3 = st.columns([1, .5, 1, .5, 1])
        with c1:
            space()
            space()
            st.image("f1.gif")
            space()
            st.subheader("Scientific Homeopathy Treatment")
            st.markdown("Reduces itching, Redness, Scaling")
            space()
            st.markdown("Natural | Safe | Effective")
            space()
            st.markdown("Gives Long Lasting Results")
            space()
            st.markdown("No side effects")
            space()
            st.markdown("Treats the root cause")
            space()
        with c2:
            space()
            space()
            st.image("f1.gif")
            space()
            st.subheader("Dermahel For Faster Sin Healing")
            st.markdown("Visible results in 5 weeks")
            space()
            st.markdown("No steroidal cream")
            space()
            st.markdown("Reduces dependency on prescription medication")
            space()
            st.markdown("Natural skin healing")
            space()
            st.markdown("Control development of new patches")
        with c3:
            space()
            space()
            st.image("f1.gif")
            space()
            st.subheader("Diet Plan for Accelerated Wound Healing")
            space()
            st.markdown("HealRight promotes natural healing methods to reduce reliance on medication.")
            space()
            st.markdown("Their program also focuses on preventing new wounds.")
            space()
            st.markdown("They recommend consulting a healthcare professional for personalized advice.")
            space()
            st.markdown("They claim visible results in 4 weeks.")
        st.divider()