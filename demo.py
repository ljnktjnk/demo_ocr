import streamlit as st
import cv2
import os
import numpy as np
from ultralytics import YOLO
import time
import glob
from PIL import ImageFont, ImageDraw, Image
from predict_rec import TextRecognizer
import yaml

with open("config.yaml") as file:
    config = yaml.safe_load(file)
text_recognizer = TextRecognizer(config)

model_detect = YOLO("./weights/best_det.pt")

fontpath = "simsun.ttc"	 
font = ImageFont.truetype(fontpath, 50)
masters = open("master.txt", "r", encoding="utf-8").readlines()
masters = [m.replace("\n", "") for m in masters]

def longest_common_substring(str1, str2):
	m = len(str1)
	n = len(str2)
	
	# Create a matrix to store the lengths of the longest common suffixes
	# dp[i][j] will store the length of the longest common suffix of str1[0...i-1] and str2[0...j-1]
	dp = [[0] * (n + 1) for _ in range(m + 1)]
	
	max_length = 0   # Length of the longest common substring
	end_index = 0	# Ending index of the longest common substring in str1
	
	for i in range(1, m + 1):
		for j in range(1, n + 1):
			if str1[i - 1] == str2[j - 1]:
				dp[i][j] = dp[i - 1][j - 1] + 1
				if dp[i][j] > max_length:
					max_length = dp[i][j]
					end_index = i
	
	# Extract the longest common substring from str1
	longest_substring = str1[end_index - max_length:end_index]
	return longest_substring, end_index - max_length


def predict(image, pre_text):
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	detect_cells = model_detect(image)

	if len(detect_cells[0].boxes) == 0:
		return None, None
	
	new_width = int(image.shape[1]*0.6)
	new_height = int(image.shape[0]*0.6)
	resize_image = cv2.resize(image, (new_width, new_height))
	result_image = np.ones((new_height, new_width*2+50, 3), dtype=np.uint8) * 255
	result_image[0:new_height, 0:new_width] = resize_image
	table_image = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255

	boxes = []
	bus_name = next_bus = ""
	x_bus = y_bus = 30
	
	# get boxes from yolo model
	for i, cell in enumerate(detect_cells[0]):
		id_cls = int(cell.boxes.cls)
		conf = cell.boxes.conf
		# if conf < 0.6:
		# 	continue
		x1, y1, x2, y2 = cell.boxes.xyxy.view(-1).tolist()
  
		x1 = int(max(0, x1))
		x2 = int(min(x2, image.shape[1]))
		y1 = int(max(0, y1))
		y2 = int(min(y2, image.shape[0]))
		crop_image = image[y1:y2, x1:x2]

		text = text_recognizer.predict_single_img(crop_image)[0][0]
		text = text.replace("\r\n", "")
  
		if id_cls==0:
			cv2.rectangle(table_image, (x1, y1), (x2, y2), (0, 0, 0), 2)
			cv2.putText(table_image, text, (int(x1+(x2-x1)/2)-10, int(y1+(y2-y1)/2)+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
		
		else:
			x_bus = x1
			y_bus = y1
			bus_name = text
	
	# Check busname 
	if bus_name in masters:
		next_bus = bus_name
	else:
		lcs, idx = longest_common_substring(pre_text, bus_name)
		end = bus_name.index(lcs) + len(lcs)
		next_bus = pre_text[0:idx] + lcs + bus_name[end:]
     
	# draw result
	img_pil = Image.fromarray(table_image)
	draw = ImageDraw.Draw(img_pil)
	draw.text((100, y_bus), "テキスト："+bus_name, font = font, fill = (0, 0, 0, 0))
	draw.text((y_bus+1000, y_bus), "次は："+next_bus, font = font, fill = (0, 0, 0, 0))
	table_image = np.array(img_pil)
	
	# combine 2 image
	result_image[0:new_height, new_width+50:new_width*2+50] = cv2.resize(table_image, (new_width, new_height))
	result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

	return result_image, next_bus

# for path in glob.glob("C:/Users/NGUYEN THI LINH/Documents/Customer/11. MRI/From Customer/Data/Det_Train_0814/*.jpg"):
#     path = "C:/Users/NGUYEN THI LINH/Documents/Customer/11. MRI/From Customer/Data/Det_Train_0814/000G0392_40500_0814.jpg"
#     image = cv2.imread(path)
#     result_image = predict(image)
#     cv2.imshow("image", image)
#     cv2.imshow("result_image", result_image)
#     cv2.waitKey()


# exit()

def main():
	st.title("VTI - Demo OCR")
	
	# Upload video file
	vid_file = None
	vid_bytes = st.file_uploader("Upload a video", type=['mp4', 'MOV'])
	if vid_bytes:
		vid_file = "uploaded_data/upload." + vid_bytes.name.split('.')[-1]
		with open(vid_file, 'wb') as out:
			out.write(vid_bytes.read())

	pre_text = ""

	if vid_file:
		# Read the video file
		video = cv2.VideoCapture(vid_file)
		output = st.empty()
		i = 0
		while video.isOpened():
			ret, frame = video.read()
			if not ret:
				break

			if i%20 == 0:
				result_image, current_text = predict(frame, pre_text)

				if result_image is None:
					continue

				pre_text = current_text
				# Display the original and processed frames
				output.image([result_image], caption=["Processed Frame"])
			
			i += 1
		
		video.release()
		
if __name__ == "__main__":
	main()
