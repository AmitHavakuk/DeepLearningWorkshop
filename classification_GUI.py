import customtkinter as ctk
from PIL import ImageTk  # For image handling with Tkinter
from tkinter import filedialog
from main_train import *
from cropper import crop_by_path


# Store embeddings of all images
CLASSES_DIR = "./classification_images"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("./models/cosinecustomSunglassesMODEL512_batch16_total20000_epochs40_lr0.0001_margin0.3.pth", map_location=device).to(device)
model.eval()
EPSILON = 0.55  # by (at least) how much you want image to be similar to 1st match to be considered a match
DISTANCE_FUNC = distances.CosineSimilarity()

######

# Store embeddings of all images
embeddings = {}

# Global variable to store the uploaded image path
uploaded_img_path = None

def create_invisible_image():
    # Create a 1x1 pixel transparent image
    transparent_image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))  # Transparent image
    return ImageTk.PhotoImage(transparent_image)


# Function to display the image in the GUI
def display_image(img_path, label):
    cropped_image = crop_by_path(img_path)

    # Convert OpenCV BGR image to RGB
    cv2_img_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    # Convert to PIL image
    img = Image.fromarray(cv2_img_rgb)

    # img = Image.open(image_path)
    img = img.resize((200, 200))
    ctk_img = ctk.CTkImage(img, size=(178, 218))
    label.configure(image=ctk_img)
    label.image = ctk_img  # Keep a reference to avoid garbage collection

# Function to calculate embeddings for all images in a folder
def calculate_embeddings(folder_path):
    global embeddings
    img_embeddings, img_paths = calculate_embeddings_parallel(folder_path)
    for idx, path in enumerate(img_paths):
        embeddings[path] = img_embeddings[idx]


# Function to upload an image from the local drive
def upload_image_from_local():
    global uploaded_img_path
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if filepath:
        uploaded_img_path = filepath
        display_image(filepath, img_label)
        calculate_and_display_closest_image()

# Function to start webcam capture
def start_webcam_capture():
    cap = cv2.VideoCapture(0)
    capture_window = ctk.CTkToplevel(app)
    capture_window.title("Capture Image")

    def capture_and_save():
        global uploaded_img_path
        ret, frame = cap.read()
        if ret:
            filename = f"./capture/captured_img.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")
            uploaded_img_path = filename
            display_image(filename, img_label)
            calculate_and_display_closest_image()
            capture_window.destroy()
            cap.release()
            cv2.destroyAllWindows()

    def show_feed():
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            ctk_img = ctk.CTkImage(img, size=(400, 300))
            webcam_label.configure(image=ctk_img)
            webcam_label.image = ctk_img
            webcam_label.after(10, show_feed)

    webcam_label = ctk.CTkLabel(capture_window, text="")
    webcam_label.pack(pady=20, expand=True, fill="both")
    show_feed()

    capture_btn = ctk.CTkButton(capture_window, text="Capture", command=capture_and_save)
    capture_btn.pack(pady=10)
    capture_window.geometry("640x480")
    capture_window.lift()
    capture_window.attributes('-topmost', True)
    capture_window.after_idle(capture_window.attributes, '-topmost', False)

# Function to calculate embedding of the uploaded image
def calculate_embedding(image_path):
    # Convert image to tensor, process, and pass through model
    cropped_image = crop_by_path(image_path)
    image = image_crop_to_tensor(cropped_image)
    tensor_img = image.to(device)
    tensor_img = torch.stack([tensor_img])
    embed = model(tensor_img)
    return embed

def calculate_embeddings_parallel(folder_path):
    img_paths = [os.path.join(folder_path, img_name) for img_name in os.listdir(folder_path)]
    cropped_images = [crop_by_path(x) for x in img_paths]
    images = [image_crop_to_tensor(x) for x in cropped_images]
    images = torch.stack(images)

    images = images.to(device)

    # Forward pass to get embeddings
    img_embeddings = model(images)  # Output shape: (batch_size, EMBEDDING_DIM)
    return img_embeddings, img_paths

# Function to calculate and display the closest image based on embeddings
def calculate_and_display_closest_image():
    global uploaded_img_path
    if uploaded_img_path is None:
        return

    uploaded_embedding = calculate_embedding(uploaded_img_path)

    closest_img = None
    closest_dist = np.NINF
    for img_name, emb in embeddings.items():
        #print("DEBUG")
        #print(uploaded_embedding.shape, emb.unsqueeze(0).shape)
        dist = DISTANCE_FUNC(uploaded_embedding, emb.unsqueeze(0))  # shapes are correct
        dist = dist.item()
        print(img_name, dist)
        if dist > closest_dist:
            closest_dist = dist
            closest_img = img_name

    # Display the closest image
    if closest_dist < EPSILON:
        invisible_img = create_invisible_image()
        closest_img_label.configure(image=invisible_img)
        closest_img_label.image = invisible_img
        closest_img_label.configure(text=f"No matches were close enough!")
    else:
        if closest_img:

            closest_img_path = closest_img
            #img = Image.open(closest_img_path)

            cropped_img = crop_by_path(closest_img_path)
            cv2_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            # Convert to PIL image
            img = Image.fromarray(cv2_img_rgb)

            # Convert the image to a Tkinter-compatible format
            img_tk = ImageTk.PhotoImage(img)

            closest_img_label.configure(image=img_tk)
            closest_img_label.image = img_tk  # Keep a reference to avoid garbage collection

            closest_img_label.configure(text=f"Closest Image: {closest_img}")

# GUI Setup
app = ctk.CTk()
app.title("Face Classification")

# Label to display the uploaded/captured image
img_label = ctk.CTkLabel(app, text="Uploaded/Captured Image", width=178, height=218)
img_label.pack(pady=10, padx=50)

# Button to calculate embeddings for all images in a folder
btn_calculate = ctk.CTkButton(app, text="Calculate Embeddings", command=lambda: calculate_embeddings(CLASSES_DIR))
btn_calculate.pack(pady=10)

# Button to upload image from local disk
btn_upload_local = ctk.CTkButton(app, text="Upload Image (Local)", command=upload_image_from_local)
btn_upload_local.pack(pady=10)

# Button to capture image from webcam
btn_upload_webcam = ctk.CTkButton(app, text="Capture Image (Webcam)", command=start_webcam_capture)
btn_upload_webcam.pack(pady=10)

closest_img_label = ctk.CTkLabel(app, text="Closest Image", width=178, height=218)
closest_img_label.pack(pady=10)


app.mainloop()
