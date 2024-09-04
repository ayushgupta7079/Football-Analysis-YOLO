from ultralytics import YOLO # type: ignore

# writing model name
model = YOLO('models/best.pt')
results = model.predict('input_videos/08fd33_4.mp4',
                        save=True)  # saving the result

print(results[0])  # printing the result of the first frame

print('--------------------------------')
for box in results[0].boxes:
    print(box)