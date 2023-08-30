import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import face_recognition
from PIL import Image

# Selfie Segmentation
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

#Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

st.title("FaceCraft Nexus")
st.subheader("Crafting Digital Expressions, One Pixel at a Time")

# Create tabs for different operations
tab1, tab2, tab3, tab4 = st.tabs(
    ["About Us", "Background Changer", "Face Detection", "Face Recognition"]
)

# About Us
with tab1:
    # Add an image below the subtitle
    image_path = "FaceCraft Nexus Logo.jpeg"
    st.image(image_path)

    st.markdown("## About Us")

    st.write("Welcome to FaceCraft Nexus, your gateway to the world of digital artistry and face-related innovations. We are a passionate team dedicated to harnessing the power of technology to create captivating visual experiences that transcend boundaries.\n")

    st.write("At FaceCraft Nexus, we believe in the fusion of creativity and cutting-edge techniques. Our platform serves as a canvas where pixels come alive, bringing your imagination to reality. From background transformations that transport you to new worlds, to the precision of face detection and recognition, we're here to redefine how you interact with images.\n")

    st.write("Our journey began with a shared fascination for the potential of Python in the realm of face-related operations. Through days of learning and exploring, we've crafted a curated collection of tools that showcase our enthusiasm for innovation. Whether you're an artist, a developer, or simply someone curious about the endless possibilities, FaceCraft Nexus is designed to spark your imagination.\n")

    st.write("Join us on this exciting journey as we uncover the artistry that resides within the digital realm. Feel free to explore our diverse array of features, from background changing to intricate face analysis. We're here to provide you with both the tools and the inspiration to push the boundaries of what's possible.\n")

    st.write("Thank you for being a part of the FaceCraft Nexus community. Together, we're crafting the future of visual expression, one pixel at a time.\n")
    
    st.write("1. Background Changer \n2. Face Detection \n3. Face Recognition")

# Background Changer
with tab2:
    st.markdown("## Background Changer")
    col1, col2 = st.columns(2)
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        with col1:
            st.markdown("### Foreground Image")
            fg_image = st.file_uploader("Upload a FOREGROUND IMAGE")
            if fg_image is not None:
                fimage = np.array(Image.open(fg_image))
                # Resize the image to a proportional size (e.g., max height or width of 400 pixels)
                max_size = 400
                height, width, _ = fimage.shape
                if height > max_size or width > max_size:
                    if height > width:
                        scale_factor = max_size / height
                    else:
                        scale_factor = max_size / width
                    new_height = int(height * scale_factor)
                    new_width = int(width * scale_factor)
                    fimage = cv2.resize(fimage, (new_width, new_height))
                st.image(fimage)
                results = selfie_segmentation.process(fimage)
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        with col2:
            if fg_image is not None:
                st.markdown("### Background Image")
                add_bg = st.selectbox(
                    "How would you like to change your Background?",
                    ("Image of your choice", "Inbuilt Image", "Colors")
                )
                if add_bg == "Image of your choice":
                    bg_image = st.file_uploader("Upload a BACKGROUND IMAGE")
                    if bg_image is not None:
                        bimage = np.array(Image.open(bg_image))
                        # Resize the image to a proportional size (e.g., max height or width of 400 pixels)
                        max_size = 400
                        height, width, _ = bimage.shape
                        if height > max_size or width > max_size:
                            if height > width:
                                scale_factor = max_size / height
                            else:
                                scale_factor = max_size / width
                            new_height = int(height * scale_factor)
                            new_width = int(width * scale_factor)
                            bimage = cv2.resize(bimage, (new_width, new_height))
                        st.image(bimage)
                        bimage = cv2.resize(bimage, (fimage.shape[1], fimage.shape[0]))
                        output_image = np.where(condition, fimage, bimage)
                        # Resize the image to a proportional size (e.g., max height or width of 400 pixels)
                        max_size = 400
                        height, width, _ = output_image.shape
                        if height > max_size or width > max_size:
                            if height > width:
                                scale_factor = max_size / height
                            else:
                                scale_factor = max_size / width
                            new_height = int(height * scale_factor)
                            new_width = int(width * scale_factor)
                            output_image = cv2.resize(output_image, (new_width, new_height))
                        st.markdown("### Final Image")
                        st.image(output_image)

                elif add_bg == "Inbuilt Image":
                    add_ib_bg = st.selectbox(
                        "Which background would you prefer?",
                        ("Beach","Library","Cherry Blossom(animated)","Spooky")
                    )
                    if add_ib_bg == "Beach":
                        bg_image = cv2.imread("Beach.jpg")
                        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_RGB2BGR)
                        if bg_image is not None: 
                            # Resize the image to a proportional size (e.g., max height or width of 400 pixels)
                            max_size = 400
                            height, width, _ = bg_image.shape
                            if height > max_size or width > max_size:
                                if height > width:
                                    scale_factor = max_size / height
                                else:
                                    scale_factor = max_size / width
                                new_height = int(height * scale_factor)
                                new_width = int(width * scale_factor)
                                bg_image = cv2.resize(bg_image, (new_width, new_height))
                            st.image(bg_image)
                            bimage = cv2.resize(bg_image, (fimage.shape[1], fimage.shape[0]))
                            output_image = np.where(condition, fimage, bimage)
                            # Resize the image to a proportional size (e.g., max height or width of 400 pixels)
                            max_size = 400
                            height, width, _ = output_image.shape
                            if height > max_size or width > max_size:
                                if height > width:
                                    scale_factor = max_size / height
                                else:
                                    scale_factor = max_size / width
                                new_height = int(height * scale_factor)
                                new_width = int(width * scale_factor)
                                output_image = cv2.resize(output_image, (new_width, new_height))
                            st.image(output_image)    
                    elif add_ib_bg == "Library":
                        bg_image = cv2.imread("Library.jpg")
                        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_RGB2BGR)
                        if bg_image is not None: 
                            # Resize the image to a proportional size (e.g., max height or width of 400 pixels)
                            max_size = 400
                            height, width, _ = bg_image.shape
                            if height > max_size or width > max_size:
                                if height > width:
                                    scale_factor = max_size / height
                                else:
                                    scale_factor = max_size / width
                                new_height = int(height * scale_factor)
                                new_width = int(width * scale_factor)
                                bg_image = cv2.resize(bg_image, (new_width, new_height))
                            st.image(bg_image)
                            bimage = cv2.resize(bg_image, (fimage.shape[1], fimage.shape[0]))
                            output_image = np.where(condition, fimage, bimage)
                            # Resize the image to a proportional size (e.g., max height or width of 400 pixels)
                            max_size = 400
                            height, width, _ = output_image.shape
                            if height > max_size or width > max_size:
                                if height > width:
                                    scale_factor = max_size / height
                                else:
                                    scale_factor = max_size / width
                                new_height = int(height * scale_factor)
                                new_width = int(width * scale_factor)
                                output_image = cv2.resize(output_image, (new_width, new_height))
                            st.image(output_image)
                    elif add_ib_bg == "Cherry Blossom(animated)":
                        bg_image = cv2.imread("Cherry Blossom.jpg")
                        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_RGB2BGR)
                        if bg_image is not None: 
                            # Resize the image to a proportional size (e.g., max height or width of 400 pixels)
                            max_size = 400
                            height, width, _ = bg_image.shape
                            if height > max_size or width > max_size:
                                if height > width:
                                    scale_factor = max_size / height
                                else:
                                    scale_factor = max_size / width
                                new_height = int(height * scale_factor)
                                new_width = int(width * scale_factor)
                                bg_image = cv2.resize(bg_image, (new_width, new_height))
                            st.image(bg_image)
                            bimage = cv2.resize(bg_image, (fimage.shape[1], fimage.shape[0]))
                            output_image = np.where(condition, fimage, bimage)
                            # Resize the image to a proportional size (e.g., max height or width of 400 pixels)
                            max_size = 400
                            height, width, _ = output_image.shape
                            if height > max_size or width > max_size:
                                if height > width:
                                    scale_factor = max_size / height
                                else:
                                    scale_factor = max_size / width
                                new_height = int(height * scale_factor)
                                new_width = int(width * scale_factor)
                                output_image = cv2.resize(output_image, (new_width, new_height))
                            st.image(output_image)   
                    elif add_ib_bg == "Spooky":
                        bg_image = cv2.imread("Horror.jpg")
                        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_RGB2BGR)
                        if bg_image is not None: 
                            # Resize the image to a proportional size (e.g., max height or width of 400 pixels)
                            max_size = 400
                            height, width, _ = bg_image.shape
                            if height > max_size or width > max_size:
                                if height > width:
                                    scale_factor = max_size / height
                                else:
                                    scale_factor = max_size / width
                                new_height = int(height * scale_factor)
                                new_width = int(width * scale_factor)
                                bg_image = cv2.resize(bg_image, (new_width, new_height))
                            st.image(bg_image)
                            bimage = cv2.resize(bg_image, (fimage.shape[1], fimage.shape[0]))
                            output_image = np.where(condition, fimage, bimage)
                            # Resize the image to a proportional size (e.g., max height or width of 400 pixels)
                            max_size = 400
                            height, width, _ = output_image.shape
                            if height > max_size or width > max_size:
                                if height > width:
                                    scale_factor = max_size / height
                                else:
                                    scale_factor = max_size / width
                                new_height = int(height * scale_factor)
                                new_width = int(width * scale_factor)
                                output_image = cv2.resize(output_image, (new_width, new_height))
                            st.image(output_image)
                    else:
                        st.write("Choose some other option")

                elif add_bg == "Colors":
                        add_c_bg = st.selectbox(
                            "Choose a color as a background",
                            ("Red","Green","Blue","Gray")
                        )
                        if add_c_bg == "Red":
                            BG_COLOR = (255,0,0)
                            bg_image = np.zeros(fimage.shape, dtype=np.uint8)
                            bg_image[:] = BG_COLOR
                            output_image = np.where(condition, fimage, bg_image)
                            # Resize the image to a proportional size (e.g., max height or width of 400 pixels)
                            max_size = 400
                            height, width, _ = output_image.shape
                            if height > max_size or width > max_size:
                                if height > width:
                                    scale_factor = max_size / height
                                else:
                                    scale_factor = max_size / width
                                new_height = int(height * scale_factor)
                                new_width = int(width * scale_factor)
                                output_image = cv2.resize(output_image, (new_width, new_height))
                            st.image(output_image)
                        elif add_c_bg == "Green":
                            BG_COLOR = (0,255,0)
                            bg_image = np.zeros(fimage.shape, dtype=np.uint8)
                            bg_image[:] = BG_COLOR
                            output_image = np.where(condition, fimage, bg_image)
                            # Resize the image to a proportional size (e.g., max height or width of 400 pixels)
                            max_size = 400
                            height, width, _ = output_image.shape
                            if height > max_size or width > max_size:
                                if height > width:
                                    scale_factor = max_size / height
                                else:
                                    scale_factor = max_size / width
                                new_height = int(height * scale_factor)
                                new_width = int(width * scale_factor)
                                output_image = cv2.resize(output_image, (new_width, new_height))
                            st.image(output_image)
                        elif add_c_bg == "Blue":
                            BG_COLOR = (0,0,255)
                            bg_image = np.zeros(fimage.shape, dtype=np.uint8)
                            bg_image[:] = BG_COLOR
                            output_image = np.where(condition, fimage, bg_image)
                            # Resize the image to a proportional size (e.g., max height or width of 400 pixels)
                            max_size = 400
                            height, width, _ = output_image.shape
                            if height > max_size or width > max_size:
                                if height > width:
                                    scale_factor = max_size / height
                                else:
                                    scale_factor = max_size / width
                                new_height = int(height * scale_factor)
                                new_width = int(width * scale_factor)
                                output_image = cv2.resize(output_image, (new_width, new_height))
                            st.image(output_image)
                        elif add_c_bg == "Gray":
                            BG_COLOR = (192,192,192)
                            bg_image = np.zeros(fimage.shape, dtype=np.uint8)
                            bg_image[:] = BG_COLOR
                            output_image = np.where(condition, fimage, bg_image)
                            # Resize the image to a proportional size (e.g., max height or width of 400 pixels)
                            max_size = 400
                            height, width, _ = output_image.shape
                            if height > max_size or width > max_size:
                                if height > width:
                                    scale_factor = max_size / height
                                else:
                                    scale_factor = max_size / width
                                new_height = int(height * scale_factor)
                                new_width = int(width * scale_factor)
                                output_image = cv2.resize(output_image, (new_width, new_height))
                            st.image(output_image)

# Face Detection
with tab3:
    st.markdown("## Face Detection")
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
        st.markdown("### Upload Image")
        fd_image = st.file_uploader("Upload a FOREGROUND IMAGE", key="foreground_image")
        if fd_image is not None:
            fdimage = np.array(Image.open(fd_image))
            # Resize the image to a proportional size (e.g., max height or width of 400 pixels)
            max_size = 400
            height, width, _ = fdimage.shape
            if height > max_size or width > max_size:
                if height > width:
                    scale_factor = max_size / height
                else:
                    scale_factor = max_size / width
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                fdimage = cv2.resize(fdimage, (new_width, new_height))
            st.image(fdimage)
            results = face_detection.process(fdimage)
            for landmark in results.detections:
                mp_drawing.draw_detection(fdimage, landmark)
            # Resize the image to a proportional size (e.g., max height or width of 400 pixels)
            max_size = 400
            height, width, _ = fdimage.shape
            if height > max_size or width > max_size:
                if height > width:
                    scale_factor = max_size / height
                else:
                    scale_factor = max_size / width
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                fdimage = cv2.resize(fdimage, (new_width, new_height))
            st.markdown("### Final Image")
            st.image(fdimage)

# Face Recognition
with tab4:
    st.markdown("## Face Recognition")
    st.markdown("### Upload TWO IMAGES")

    col1, col2 = st.columns(2)

    # Training Image
    with col1:
        st.markdown("### Training Image")
        train_image = st.file_uploader("Upload an image to train", key="train_image")
        if train_image is not None:
            train_image_np = np.array(Image.open(train_image))
            st.image(train_image_np)
            image_encodings_train = face_recognition.face_encodings(train_image_np)[0]

    # Testing Image
    with col2:
        st.markdown("### Testing Image")
        detect_image = st.file_uploader("Upload an image to test", key="test_image")
        if detect_image is not None:
            test_image_np = np.array(Image.open(detect_image))
            st.image(test_image_np)
            image_encodings_test = face_recognition.face_encodings(test_image_np)[0]
            image_location_test = face_recognition.face_locations(test_image_np)

            results = face_recognition.compare_faces([image_encodings_test], image_encodings_train)
            face_distance = face_recognition.face_distance([image_encodings_train], image_encodings_test)
            accuracy_threshold = 0.6  # Adjust this threshold as needed
            match = results[0] and face_distance[0] <= accuracy_threshold

            if match:
                output_image = test_image_np.copy()
                for (top, right, bottom, left) in image_location_test:
                    accuracy_percentage = (1 - face_distance[0]) * 100
                    cv2.rectangle(output_image, (left, top), (right, bottom), (0, 255, 0), 4)
                    cv2.putText(output_image, f"Accuracy: {accuracy_percentage:.2f}%", (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                st.image(output_image)
            else:
                output_image = test_image_np.copy()
                for (top, right, bottom, left) in image_location_test:
                    accuracy_percentage = (1 - face_distance[0]) * 100
                    cv2.rectangle(output_image, (left, top), (right, bottom), (255, 0, 0), 4)
                    cv2.putText(output_image, f"Accuracy: {accuracy_percentage:.2f}%", (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                st.image(output_image)
                st.write("Both faces don't match")
