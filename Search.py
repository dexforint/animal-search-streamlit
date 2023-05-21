import streamlit as st


title = 'Animal Search Classification'
st.set_page_config(
    page_title=title,
    page_icon="👋",
)


with st.spinner('Инициализация нейронной сети...'):
    import numpy as np
    import torch
    # from transformers import OwlViTProcessor, OwlViTForObjectDetection
    from PIL import Image
    import cv2
    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ####


    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", load_in_8bit=True)
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = model.to(device)
    model.eval()

    ####################

def predict(image, query, score_threshold=0.05):
    inputs = processor(text=[query], images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = torch.max(outputs["logits"][0], dim=-1)
    scores = torch.sigmoid(logits.values).cpu().detach().numpy()

    # labels = logits.indices.cpu().detach().numpy()
    boxes = outputs["pred_boxes"][0].cpu().detach().numpy()

    predictions = []
    H, W, _ = image.shape

    for i, (box, score) in enumerate(zip(boxes, scores)):
        if score < score_threshold:
            continue

        x, y, w, h = box
        x = max(min(x * W, W), 0)
        w = w * W
        y = max(min(y * H, H), 0)
        h = h * H

        x1 = x - w / 2
        x2 = x + w / 2
        y1 = y - h / 2
        y2 = y + h / 2

        x1 = int(max(x1, 0))
        x2 = int(min(x2, W))
        y1 = int(max(y1, 0))
        y2 = int(min(y2, H))
        thickness = (H + W) // 600
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), thickness)


    return image


st.markdown(f"# {title} :sunglasses:")
st.success("## Команда: ML Rocks")
st.markdown("#")

st.divider()

st.markdown("### Обнаруживаем животных")

query  = st.text_input("Название животного на английском")
img_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
threshold  = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if (img_file is not None) and len(query) > 0:
    img = Image.open(img_file).convert('RGB') 
    img = np.array(img)

    with st.spinner('Обработка изображения'):
        img = predict(img, query, threshold)
    
    st.image(img, caption=f"Результат распознавания")
    st.markdown(f"**По каким либо вопросам**: https://t.me/dl_hello")


st.divider()
