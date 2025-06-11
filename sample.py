import os,cv2,torch,tempfile,streamlit as st,pandas as pd
from PIL import Image
from transformers import BlipProcessor,BlipForConditionalGeneration

BASE_DIR="./"
VIDEO_DIR=os.path.join(BASE_DIR,"videos")
HUMAN_LABELS_PATH=os.path.join(BASE_DIR,"human.csv")

@st.cache_resource
def load_blip_model():
 processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
 model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cpu")
 return processor,model

processor,model=load_blip_model()

def extract_middle_frame(path):
 cap=cv2.VideoCapture(path)
 total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 cap.set(cv2.CAP_PROP_POS_FRAMES,total//2)
 ret,frame=cap.read();cap.release()
 return Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)) if ret else None

def caption_image(image):
 inputs=processor(image,return_tensors="pt").to("cpu")
 with torch.no_grad():out=model.generate(**inputs)
 return processor.decode(out[0],skip_special_tokens=True)

def load_human_labels(path):
 try:
  df=pd.read_csv(path)
  required={'video','caption1','caption2','caption3','caption4'}
  if not required.issubset(df.columns):return None
  return df
 except:return None

def pipeline_from_directory():
 st.header("Videos from Local Directory")
 files=[f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4','.mov','.avi'))]
 results=[]
 for name in files:
  path=os.path.join(VIDEO_DIR,name)
  img=extract_middle_frame(path)
  if img:
   ai=caption_image(img)
   row=human_df[human_df['video']==name]
   if not row.empty:
    c1,c2,c3=st.columns(3)
    with c1:st.image(img,caption="🎞️ Middle Frame",use_column_width=True)
    with c2:
     st.markdown("**🧑 Human Captions**")
     for i in range(1,5):st.info(row.iloc[0][f"caption{i}"])
    with c3:
     st.markdown("**🤖 AI Caption**")
     st.success(ai)
    results.append({"video":name,"ai_caption":ai,"caption1":row.iloc[0]["caption1"],"caption2":row.iloc[0]["caption2"],"caption3":row.iloc[0]["caption3"],"caption4":row.iloc[0]["caption4"]})
 if results:
  df=pd.DataFrame(results)
  csv=df.to_csv(index=False).encode("utf-8")
  st.download_button("📥 Download CSV",csv,"directory_video_captions.csv","text/csv")

def pipeline_from_upload():
 st.header("Upload and Caption Videos")
 uploads=st.file_uploader("Upload video files",type=["mp4","mov","avi"],accept_multiple_files=True)
 if uploads:
  results=[]
  for file in uploads:
   st.markdown(f"---\n### 📁 {file.name}")
   with tempfile.NamedTemporaryFile(delete=False,suffix=".mp4") as tmp:
    tmp.write(file.read());tmp_path=tmp.name
   img=extract_middle_frame(tmp_path)
   if img:
    ai=caption_image(img)
    st.image(img,caption="🎞️ Middle Frame",use_column_width=True)
    st.success(f"🤖 AI Caption: {ai}")
    results.append({"video":file.name,"ai_caption":ai})
   else:st.error("Could not extract frame.")
  if results:
   df=pd.DataFrame(results)
   csv=df.to_csv(index=False).encode("utf-8")
   st.download_button("📥 Download CSV",csv,"uploaded_video_captions.csv","text/csv")

st.title("🎬 Video-to-Text Captioning Pipeline")
opt=st.sidebar.selectbox("Select Section",["📁 Caption Videos from Directory","📤 Upload and Caption New Videos"])

human_df=load_human_labels(HUMAN_LABELS_PATH)
if human_df is None and opt=="📁 Caption Videos from Directory":
 st.error("Invalid or missing 'human.csv'. It must have columns: video, caption1-4.")
else:
 if opt=="📁 Caption Videos from Directory":pipeline_from_directory()
 if opt=="📤 Upload and Caption New Videos":pipeline_from_upload()
