import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import face_recognition
from PIL import Image, ExifTags

# Fix Image Orientation
def fix_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image

# Selfie Segmentation
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

#Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

st.title("FaceCraft Nexus")
st.subheader("Crafting Digital Expressions, One Pixel at a Time")

# Create tabs for different operations with custom CSS
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:20px;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

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
                fimage = np.array(fix_image_orientation(Image.open(fg_image)))
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
                        bimage = np.array(fix_image_orientation(Image.open(bg_image)))
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
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Upload Image")
            fd_image = st.file_uploader("Upload an image to detect face", key="image")

            if fd_image is not None:
                fdimage = np.array(fix_image_orientation(Image.open(fd_image)))
                max_size = 400
                height, width, _ = fdimage.shape
                if height > max_size or width > max_size:
                    scale_factor = max_size / max(height, width)
                    new_height = int(height * scale_factor)
                    new_width = int(width * scale_factor)
                    fdimage = cv2.resize(fdimage, (new_width, new_height))
                st.image(fdimage)

        with col2:
            if fd_image is not None:
                results = face_detection.process(fdimage)
                for landmark in results.detections:
                    mp_drawing.draw_detection(fdimage, landmark, mp_drawing.DrawingSpec(color=(0, 255, 0), circle_radius=0))
                max_size = 400
                height, width, _ = fdimage.shape
                if height > max_size or width > max_size:
                    scale_factor = max_size / max(height, width)
                    new_height = int(height * scale_factor)
                    new_width = int(width * scale_factor)
                    fdimage = cv2.resize(fdimage, (new_width, new_height))
                st.markdown("### Final Image")
                st.image(fdimage)

# Face Recognition
with tab4:
    st.markdown("## Face Recognition")

    # Set accuracy threshold for face recognition
    accuracy_threshold = 0.6

    # Set Column
    col1, col2 = st.columns(2)

    # Training Image
    with col1:
        st.markdown("### Upload Training Images")
        train_images = st.file_uploader("Upload training images", accept_multiple_files=True, key="train_images")
        train_names = [st.text_input(f"Name for Image {i+1}", key=f"name_{i}") for i in range(len(train_images))]
        if train_images:
            # Display training images in a slideshow
            slideshow_index = st.session_state.get("slideshow_index", 0)

            prev_button_disabled = slideshow_index == 0
            next_button_disabled = slideshow_index == len(train_images) - 1

            prev_button, next_button = st.columns(2)
            if prev_button.button("Previous", key="prev_button", disabled=prev_button_disabled):
                slideshow_index = max(slideshow_index - 1, 0)

            if next_button.button("Next", key="next_button", disabled=next_button_disabled):
                slideshow_index = min(slideshow_index + 1, len(train_images) - 1)

            st.image(train_images[slideshow_index], caption=train_names[slideshow_index])

            st.session_state.slideshow_index = slideshow_index

    # Testing Image
    with col2:
        st.markdown("### Upload Testing Image")
        test_image = st.file_uploader("Upload a testing image", key="test_image")
        # Display testing images in a slideshow
        if test_image is not None:
            st.image(test_image, caption="Uploaded Testing Image")

    if train_images and test_image:
        # Load training images and encode faces
        train_image_nps = [np.array(fix_image_orientation(Image.open(image))) for image in train_images]
        train_encodings = [face_recognition.face_encodings(train_image_np)[0] for train_image_np in train_image_nps]

        # Load testing image and detect faces
        test_image_np = np.array(fix_image_orientation(Image.open(test_image)))
        test_face_locations = face_recognition.face_locations(test_image_np)
        test_face_encodings = face_recognition.face_encodings(test_image_np, test_face_locations)

        st.markdown("### Results")

        # Perform face recognition on each face in the testing image
        for i, test_face_encoding in enumerate(test_face_encodings):
            matches = face_recognition.compare_faces(train_encodings, test_face_encoding, tolerance=accuracy_threshold)
            face_distances = face_recognition.face_distance(train_encodings, test_face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = train_names[best_match_index]
                accuracy = (1 - face_distances[best_match_index]) * 100
                if accuracy >= accuracy_threshold * 100:
                    color = (0, 255, 0)  # Green for matches
                else:
                    color = (255, 255, 0)  # Yellow for borderline matches
            else:
                name = "Unknown"
                accuracy = 0
                color = (255, 0, 0)  # Red for no matches

            top, right, bottom, left = test_face_locations[i]
            cv2.rectangle(test_image_np, (left, top), (right, bottom), color, 4)
            cv2.putText(test_image_np, f"{name} ({accuracy:.2f}%)", (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 2, color, 2)

        # Resize the displayed testing image
        max_size = 400
        height, width, _ = test_image_np.shape
        if height > max_size or width > max_size:
            if height > width:
                scale_factor = max_size / height
            else:
                scale_factor = max_size / width
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            resized_test_image = cv2.resize(test_image_np, (new_width, new_height))
        else:
            resized_test_image = test_image_np
            
        st.image(test_image_np, caption="Testing Image with Recognition Results")

    # Reload Button
    reload_button = st.button("Wanna Continue?")

    # Reload the page if the reload button is clicked
    if reload_button:
        st.experimental_rerun()
