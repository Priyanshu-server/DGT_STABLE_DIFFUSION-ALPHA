from datasets import load_dataset
import uuid

def get_dreambooth_data_hugging_face(dataset_name,data_dir):
    data = load_dataset(dataset_name)
    for i,item in enumerate(data['train']):
        img = item['image']
        img.save(f"{data_dir}/{str(i) + str(uuid.uuid4())}.jpg")
    print(f"Data Download Successful from Hugging Face to {data_dir}")
