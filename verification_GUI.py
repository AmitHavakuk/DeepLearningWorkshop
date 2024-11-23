import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image
import torch
from main_train import *
from cropper import crop_by_path

ctk.set_appearance_mode("dark")
window = ctk.CTk()
window.title("Face Verification")
window.geometry("650x700")

img1_path = None
img2_path = None

# Labels to display images
img1_label = ctk.CTkLabel(window, text="       Image 1 Preview", anchor="center")
img1_label.place(x=100, y=100)
img2_label = ctk.CTkLabel(window, text="       Image 2 Preview", anchor="center")
img2_label.place(x=400, y=100)


THRESHOLD = 0.55  # Recommended 0.55

model_sunglasses = torch.load("./models/cosinecustomSunglassesMODEL512_batch16_total20000_epochs40_lr0.0001_margin0.3.pth", map_location=device).to(device)
model_sunglasses.eval()
model_mask = torch.load("./models/cosinecustomMaskMODEL512_batch16_total20000_epochs30_lr0.001_margin0.2.pth", map_location=device).to(device)
model_mask.eval()
model_combined = torch.load("./models/cosinecustomCombinedMODEL512_batch16_total20000_epochs15_lr0.0001_margin0.2.pth", map_location=device).to(device)
model_combined.eval()

model = model_sunglasses
options = {"model_sunglasses" : model_sunglasses, "model_mask" : model_mask, "model_combined" : model_combined}

# Load the pre-trained cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def model_check_similarity(model, img1_path, img2_path):
    cropped_images = [crop_by_path(img1_path), crop_by_path(img2_path)]
    images = [image_crop_to_tensor(x) for x in cropped_images]
    images = torch.stack(images)

    images = images.to(device)

    # Forward pass to get embeddings
    embeddings = model(images)  # Output shape: (batch_size, EMBEDDING_DIM)
    # print("DEBUG")
    # print(embeddings.shape)
    # print(embeddings[0].shape)
    # print(embeddings[0].unsqueeze(0).shape)

    dist = DISTANCE_FUNC(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(
        0))  # hope this does cosine similarity
    dist = dist.item()

    same = dist > THRESHOLD

    return same, dist, THRESHOLD


# Function to display image on GUI using CTkImage
def display_image(image_path, label):
    cropped_image = crop_by_path(image_path)

    # Convert OpenCV BGR image to RGB
    cv2_img_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    # Convert to PIL image
    img = Image.fromarray(cv2_img_rgb)

    #img = Image.open(image_path)
    img = img.resize((200, 200))
    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 200))
    label.configure(image=ctk_img, text="")
    label.image = ctk_img  # Keep reference to avoid garbage collection


# Function to capture an image from webcam
def start_webcam_capture(cam_id):
    cap = cv2.VideoCapture(0)
    capture_window = ctk.CTkToplevel(window)
    capture_window.title("Capture Image")

    def capture_and_save():
        ret, frame = cap.read()
        if ret:
            filename = f"./capture/img{cam_id}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")
            display_image(filename, img1_label if cam_id == 1 else img2_label)
            update_image_path(cam_id, filename)
            capture_window.destroy()
            cap.release()
            cv2.destroyAllWindows()

    def show_feed():
        ret, frame = cap.read()
        if ret:
            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                  minNeighbors=5,
                                                  minSize=(30, 30))

            # Draw rectangles with padding around the detected faces
            for (x, y, w, h) in faces:
                padding = 50
                x_start = max(x - padding, 0)
                y_start = max(y - padding, 0)
                x_end = min(x + w + padding, frame.shape[1])
                y_end = min(y + h + padding, frame.shape[0])
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end),
                              (0, 255, 0), 2)

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to Image for CTkImage
            img = Image.fromarray(frame_rgb)
            #print("webcam dims: ", (webcam_label.winfo_width(), webcam_label.winfo_height()))
            ctk_img = ctk.CTkImage(img, size=((300,300)))
            webcam_label.configure(image=ctk_img)
            webcam_label.image = ctk_img  # Keep a reference to avoid garbage collection
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # img = Image.fromarray(frame)
            # ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
            #                        size=(400, 300))
            # webcam_label.configure(image=ctk_img)
            # webcam_label.image = ctk_img
            webcam_label.after(10, show_feed)

    # Set up the video feed label
    webcam_label = ctk.CTkLabel(capture_window, text="")
    webcam_label.pack(pady=20, expand=True, fill="both")

    show_feed()

    capture_btn = ctk.CTkButton(capture_window, text="Capture",
                                command=capture_and_save)
    capture_btn.pack(pady=10)

    capture_window.title("Webcam Capture")

    # Set the size of the capture window
    capture_window.geometry("640x480")

    #Bring window to front
    capture_window.lift()
    capture_window.attributes('-topmost', True)
    capture_window.after_idle(capture_window.attributes, '-topmost', False)


# Functions to upload images
def upload_image(cam_id):
    filepath = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if filepath:
        if cam_id == 1:
            global img1_path
            img1_path = filepath
            display_image(filepath, img1_label)
        else:
            global img2_path
            img2_path = filepath
            display_image(filepath, img2_label)
        update_image_path(cam_id, filepath)


# Update the global image paths and trigger similarity check
def update_image_path(cam_id, filepath):
    global img1_path, img2_path
    if cam_id == 1:
        img1_path = filepath
    else:
        img2_path = filepath

    check_similarity()


# Function to check similarity if both images are taken
def check_similarity():
    if img1_path and img2_path:
        res, dist, thresh = model_check_similarity(model, img1_path, img2_path)

        similarity_label.configure(
            text=f"Same person: {res}\nSimilarity: {dist}\nThreshold: {thresh}")


# Function to update the model variable when a new option is selected
def update_model(selected_value):
    global model
    model = options[selected_value]  # Update the global model variable
    check_similarity()
    print(f"Model value changed to: {selected_value}")  # Feedback in console



option_menu = ctk.CTkOptionMenu(
    window,
    values=list(options.keys()),
    command=update_model  # Call this function when an option is selected
)
option_menu.pack(pady=20)

# Set the default value of the OptionMenu to match the initial model value
option_menu.set("model_sunglasses")

# Buttons for uploading and capturing images
upload_btn1 = ctk.CTkButton(window, text="Upload Image 1",
                            command=lambda: upload_image(1))
upload_btn1.place(x=100, y=350)

capture_btn1 = ctk.CTkButton(window, text="Capture Image 1",
                             command=lambda: start_webcam_capture(1))
capture_btn1.place(x=100, y=400)

upload_btn2 = ctk.CTkButton(window, text="Upload Image 2",
                            command=lambda: upload_image(2))
upload_btn2.place(x=400, y=350)

capture_btn2 = ctk.CTkButton(window, text="Capture Image 2",
                             command=lambda: start_webcam_capture(2))
capture_btn2.place(x=400, y=400)

# Label to display similarity result
similarity_label = ctk.CTkLabel(window, text="Similarity: Not checked")
similarity_label.place(x=250, y=500)

# Bring window to front
window.lift()
window.attributes('-topmost',True)
window.after_idle(window.attributes,'-topmost',False)

window.mainloop()
