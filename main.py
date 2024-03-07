import mediapipe as mp
import sqlite3
import cv2 as cv
import tkinter as tk
from tkinter import messagebox
import threading
import os
from openai import OpenAI

client = OpenAI(
    api_key='insert api key'
)

system_message = ("You are going to act as a helpful dance instructor. You must give clear and simple feedback "
                  "regarding what a student did wrong in a specific dance movement. This movement is a way for a "
                  "dancer to go from standing to lying on their side; it consists of five steps, beginning from a "
                  "neutral standing position: crouch, place your first hand (right or left) on the ground to the "
                  "outside of that same-side leg, lower the same-side knee and hip to the floor, place your other "
                  "hand on the floor front of you, place your first hand out further away from you, lowering the "
                  "elbow of that arm to the floor in order to lay on your side completely. You will be provided with "
                  "a message describing simply what the student messed up. You will then speak as if you are "
                  "addressing the student's classroom teacher and provide corrective, yet kind and encouraging "
                  "feedback.")

white = (255, 255, 255)
red = (0, 0, 255)

data_file = r"C:\Users\s706313\PycharmProjects\pythonProject\Data"


class ErrorChecker:
    def __init__(self, ground_threshold, y):
        self.ground_threshold = ground_threshold
        self.y = y

    def ec_one(self, lk_y, lw_y, lh_y):  # hand before crouching
        if ((self.ground_threshold - lw_y) / self.y < 0.15) and ((lk_y - lh_y) / self.y > 0.15):
            return True
        else:
            return False

    def ec_two(self, lh_y, lw_y):  # hip before first hand
        if ((self.ground_threshold - lh_y) / self.y < 0.15) and ((self.ground_threshold - lw_y) / self.y > 0.15):
            return True
        else:
            return False

    def ec_three(self, lk_y, lw_y):  # knee before first hand
        if ((self.ground_threshold - lk_y) / self.y < 0.10) and ((self.ground_threshold - lw_y) / self.y > 0.15):
            return True
        else:
            return False

    def ec_four(self, le_y, lh_y, lk_y):  # elbow before hip and knee
        if ((self.ground_threshold - le_y) / self.y < 0.10) and (
                ((self.ground_threshold - lh_y) / self.y > 0.15) and ((self.ground_threshold - lk_y) / self.y > 0.15)):
            return True
        else:
            return False

    def ec_five(self, rw_y, lh_y, lk_y):  # second hand before hip and knee
        if ((self.ground_threshold - rw_y) / self.y < 0.15) and (
                ((self.ground_threshold - lh_y) > 0.15) and ((self.ground_threshold - lk_y) / self.y > 0.15)):
            return True
        else:
            return False

    def ec_six(self, rw_y, le_y):  # elbow before second hand
        if ((self.ground_threshold - le_y) / self.y < 0.10) and ((self.ground_threshold - rw_y) / self.y > 0.15):
            return True
        else:
            return False


class ErrorWriter:
    def __init__(self, df, stud_name, sm):
        self.df = df
        self.stud_name = stud_name
        self.system_message = sm

    def chat(self, error_message):
        output = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": "The student placed their hip and knee on the floor before their first "
                                            "hand."},
                {"role": "assistant", "content": "It seems the student lowered their hip and knee to the floor before "
                                                 "placing their first hand down. This is easily fixable! Remind them "
                                                 "that, in order to go from standing to lying down smoothly, "
                                                 "it is necessary to brace the initial impact with their hand, "
                                                 "not their hip or knee."},
                {"role": "user", "content": error_message}
            ]

        )
        return output.choices[0].message.content

    def ew_one(self, stud_name):
        error_message = "Student placed their hand down before crouching"
        with open(self.df, 'r') as f:
            content = f.read()  # string of .txt file
        target = "Student: " + stud_name
        length = len(target)
        index = content.find(target)
        if index != -1:  # check if target was found
            with open(self.df, 'w') as f:
                new_content = content[:index + length] + "\n-" + error_message + content[
                                                                                 index + length:]  # insert error msg
                f.write(new_content)
                messagebox.showinfo("Feedback", self.chat(error_message))

        else:
            messagebox.showerror("Error", "Student Not Found")

    def ew_two(self, stud_name):
        error_message = "Student placed their hip down before their hand"
        with open(self.df, 'r') as f:
            content = f.read()
        target = "Student: " + stud_name
        length = len(target)
        index = content.find(target)
        if index != -1:
            with open(self.df, 'w') as f:
                new_content = content[:index + length] + "\n-" + error_message + content[index + length:]
                f.write(new_content)
                messagebox.showinfo("Feedback", self.chat(error_message))

        else:
            messagebox.showerror("Error", "Student Not Found")

    def ew_three(self, stud_name):
        error_message = "Student placed their knee down before their hand"
        with open(self.df, 'r') as f:
            content = f.read()
        target = "Student: " + stud_name
        length = len(target)
        index = content.find(target)
        if index != -1:
            with open(self.df, 'w') as f:
                new_content = content[
                              :index + length] + "\n-" + error_message + content[index + length:]
                f.write(new_content)
                messagebox.showinfo("Feedback", self.chat(error_message))
        else:
            messagebox.showerror("Error", "Student Not Found")

    def ew_four(self, stud_name):
        error_message = "Student placed their elbow down before their hip and knee"
        with open(self.df, 'r') as f:
            content = f.read()
        target = "Student: " + stud_name
        length = len(target)
        index = content.find(target)
        if index != -1:
            with open(self.df, 'w') as f:
                new_content = content[
                              :index + length] + "\n-" + error_message + content[index + length:]
                f.write(new_content)
                messagebox.showinfo("Feedback", self.chat(error_message))
        else:
            messagebox.showerror("Error", "Student Not Found")

    def ew_five(self, stud_name):
        error_message = "Student placed their second hand down before their hip and knee"
        with open(self.df, 'r') as f:
            content = f.read()
        target = "Student: " + stud_name
        length = len(target)
        index = content.find(target)
        if index != -1:
            with open(self.df, 'w') as f:
                new_content = content[
                              :index + length] + "\n-" + error_message + content[index + length:]
                f.write(new_content)
                messagebox.showinfo("Feedback", self.chat(error_message))
        else:
            messagebox.showerror("Error", "Student Not Found")

    def ew_six(self, stud_name):
        error_message = "Student placed their elbow down before their second hand"
        with open(self.df, 'r') as f:
            content = f.read()
        target = "Student: " + stud_name
        length = len(target)
        index = content.find(target)
        if index != -1:
            with open(self.df, 'w') as f:
                new_content = content[
                              :index + length] + "\n-" + error_message + content[index + length:]
                f.write(new_content)
                messagebox.showinfo("Feedback", self.chat(error_message))
        else:
            messagebox.showerror("Error", "Student Not Found")


def vid_analysis(path, df, stud_name, sm):
    ground_threshold = None
    flags = [False] * 6
    pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.8)
    capture = cv.VideoCapture(path)
    first_frame = True

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        if hasattr(result, 'pose_landmarks'):
            if result.pose_landmarks:
                left_foot = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX]
                right_foot = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX]
                left_elbow = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
                right_elbow = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
                right_knee = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
                left_knee = result.pose_landmarks.landmark[
                    mp.solutions.pose.PoseLandmark.LEFT_KNEE]  # Rename built-in mediapipe keypoint objects
                right_wrist = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
                left_wrist = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
                left_hip = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                right_hip = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

                frame = cv.resize(frame, (int(frame.shape[0] / 2), int(frame.shape[1] / 1.2)))  # Resize frame if needed
                height, width, channels = frame.shape

                lf_x = int(left_foot.x * width)
                lf_y = int(left_foot.y * height)
                rf_x = int(right_foot.x * width)
                rf_y = int(right_foot.y * height)
                le_x = int(left_elbow.x * width)
                le_y = int(left_elbow.y * height)
                re_x = int(right_elbow.x * width)
                re_y = int(right_elbow.y * height)
                lk_x = int(left_knee.x * width)
                lk_y = int(left_knee.y * height)
                rk_x = int(right_knee.x * width)
                rk_y = int(right_knee.y * height)
                lw_x = int(left_wrist.x * width)
                lw_y = int(left_wrist.y * height)
                rw_x = int(right_wrist.x * width)
                rw_y = int(right_wrist.y * height)
                lh_x = int(left_hip.x * width)
                lh_y = int(left_hip.y * height)
                rh_x = int(right_hip.x * width)
                rh_y = int(right_hip.y * height)

                cv.circle(frame, (le_x, le_y), 5, red, -1)
                cv.circle(frame, (re_x, re_y), 5, red, -1)
                cv.circle(frame, (lk_x, lk_y), 5, red, -1)
                cv.circle(frame, (rk_x, rk_y), 5, red, -1)
                cv.circle(frame, (lw_x, lw_y), 5, red, -1)
                cv.circle(frame, (rw_x, rw_y), 5, red, -1)  # Draw red keypoint circles
                cv.circle(frame, (lh_x, lh_y), 5, red, -1)
                cv.circle(frame, (rh_x, rh_y), 5, red, -1)
                cv.circle(frame, (lf_x, lf_y), 5, red, -1)
                cv.circle(frame, (rf_x, rf_y), 5, red, -1)

                if first_frame:
                    ground_threshold = (lf_y + rf_y) / 2  # Establish ground y-coord in first frame
                    first_frame = False

                checker = ErrorChecker(ground_threshold, height)
                writer = ErrorWriter(df, stud_name, sm)

                if checker.ec_one(lk_y, lw_y, lh_y):
                    if not flags[0]:
                        writer.ew_one(stud_name)
                    flags[0] = True

                if checker.ec_two(lh_y, lw_y):
                    if not flags[1]:
                        writer.ew_two(stud_name)
                    flags[1] = True

                if checker.ec_three(lk_y, lw_y):
                    if not flags[2]:
                        writer.ew_three(stud_name)
                    flags[2] = True

                if checker.ec_four(le_y, lh_y, lk_y):
                    if not flags[3]:
                        writer.ew_four(stud_name)
                    flags[3] = True

                if checker.ec_five(rw_y, lh_y, lk_y):
                    if not flags[4]:
                        writer.ew_five(stud_name)
                    flags[4] = True

                if checker.ec_six(rw_y, le_y):
                    if not flags[5]:
                        writer.ew_six(stud_name)
                    flags[5] = True

        cv.imshow("Video", frame)
        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()


def pic_analysis(path, df, stud_name, sm):
    ground_threshold = None
    flags = [False] * 6
    pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    frame = cv.imread(path)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)
    if hasattr(result, 'pose_landmarks'):
        if result.pose_landmarks:
            left_foot = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX]
            right_foot = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX]
            left_elbow = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
            right_knee = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
            left_knee = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
            right_wrist = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            left_wrist = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            left_hip = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            right_hip = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

            frame = cv.resize(frame, (500, 500))

            height, width, channels = frame.shape
            lf_x = int(left_foot.x * width)
            lf_y = int(left_foot.y * height)
            rf_x = int(right_foot.x * width)
            rf_y = int(right_foot.y * height)
            le_x = int(left_elbow.x * width)
            le_y = int(left_elbow.y * height)
            re_x = int(right_elbow.x * width)
            re_y = int(right_elbow.y * height)
            lk_x = int(left_knee.x * width)
            lk_y = int(left_knee.y * height)
            rk_x = int(right_knee.x * width)
            rk_y = int(right_knee.y * height)
            lw_x = int(left_wrist.x * width)
            lw_y = int(left_wrist.y * height)
            rw_x = int(right_wrist.x * width)
            rw_y = int(right_wrist.y * height)
            lh_x = int(left_hip.x * width)
            lh_y = int(left_hip.y * height)
            rh_x = int(right_hip.x * width)
            rh_y = int(right_hip.y * height)

            checker = ErrorChecker(ground_threshold, height)
            writer = ErrorWriter(df, stud_name, sm)

            if checker.ec_one(lk_y, lw_y, lh_y):
                if not flags[0]:
                    writer.ew_one(stud_name)
                flags[0] = True

            if checker.ec_two(lh_y, lw_y):
                if not flags[1]:
                    writer.ew_two(stud_name)
                flags[1] = True

            if checker.ec_three(lk_y, lw_y):
                if not flags[2]:
                    writer.ew_three(stud_name)
                flags[2] = True

            if checker.ec_four(le_y, lh_y, lk_y):
                if not flags[3]:
                    writer.ew_four(stud_name)
                flags[3] = True

            if checker.ec_five(rw_y, lh_y, lk_y):
                if not flags[4]:
                    writer.ew_five(stud_name)
                flags[4] = True

            if checker.ec_six(rw_y, le_y):
                if not flags[5]:
                    writer.ew_six(stud_name)
                flags[5] = True

            cv.circle(frame, (le_x, le_y), 5, red, -1)
            cv.circle(frame, (re_x, re_y), 5, red, -1)
            cv.circle(frame, (lk_x, lk_y), 5, red, -1)
            cv.circle(frame, (rk_x, rk_y), 5, red, -1)
            cv.circle(frame, (lw_x, lw_y), 5, red, -1)
            cv.circle(frame, (rw_x, rw_y), 5, red, -1)  # Draw red keypoint circles
            cv.circle(frame, (lh_x, lh_y), 5, red, -1)
            cv.circle(frame, (rh_x, rh_y), 5, red, -1)
            cv.circle(frame, (lf_x, lf_y), 5, red, -1)
            cv.circle(frame, (rf_x, rf_y), 5, red, -1)

            # Display the analyzed image
            cv.imshow("Analyzed Image", frame)
            cv.waitKey(0)
            cv.destroyAllWindows()
    else:
        print("No pose landmarks detected in the image.")


def create_sql_table():
    connection = sqlite3.connect("real.db")
    cursor = connection.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS all_students (name, videos, photos)''')
    connection.commit()
    connection.close()


create_sql_table()


class LoginScreen:
    def __init__(self, root, df, sm):
        self.sm = sm
        self.df = df
        self.root = root
        self.root.title = "Home"
        self.root.geometry("2560x1440")

        self.pass_entry = tk.Entry(root, show="*")
        self.pass_entry.grid(row=0, column=0)

        self.login_button = tk.Button(root, text="Admin Login", command=self.login)
        self.login_button.grid(row=1, column=0)

    def login(self):
        password = self.pass_entry.get()

        if password == "password":
            self.root.destroy()
            home_screen(self.df, self.sm)

        else:
            messagebox.showerror("Error", "Incorrect Password")


class HomeScreen:
    def __init__(self, root, df, sm):
        self.sm = sm
        self.df = df
        self.root = root
        self.root.title = "Home"
        self.root.geometry("2560x1440")

        self.student_page_button = tk.Button(root, text="Edit Students Page", command=self.student_switch)
        self.student_page_button.grid(row=4, column=3)

        self.funct_page_button = tk.Button(root, text="Analysis Page", command=self.funct_switch)
        self.funct_page_button.grid(row=4, column=6)

    def student_switch(self):
        self.root.destroy()
        stud_screen(self.df, self.sm)

    def funct_switch(self):
        self.root.destroy()
        funct_screen(self.df, self.sm)


class FunctionScreen:
    def __init__(self, root, df, sm):
        self.sm = sm
        self.name = None
        self.df = df
        self.root = root
        self.root.title = "Analysis"
        self.root.geometry("2560x1440")

        self.pfp_label = tk.Label(root, text="Enter Filepath for Photo")
        self.pfp_label.grid(row=0, column=0)

        self.pfp_entry = tk.Entry(root)
        self.pfp_entry.grid(row=1, column=0)

        self.pfp_button = tk.Button(root, text="Submit",
                                    command=lambda: self.pic_thread(self.pfp_entry.get(), self.name, sm))
        self.pfp_button.grid(row=2, column=0)

        self.vfp_label = tk.Label(root, text="Enter Filepath for Video")
        self.vfp_label.grid(row=0, column=3)

        self.stud_label = tk.Label(root, text="Which Student is This?")
        self.stud_label.grid(row=2, column=1)

        self.vfp_entry = tk.Entry(root)
        self.vfp_entry.grid(row=1, column=3)

        self.stud_entry = tk.Entry(root)
        self.stud_entry.grid(row=3, column=1)

        self.vfp_button = tk.Button(root, text="Submit",
                                    command=lambda: self.vid_thread(self.vfp_entry.get(), self.stud_entry.get(), sm))
        self.vfp_button.grid(row=2, column=3)

        self.home_button = tk.Button(root, text="Home", command=self.home_switch)
        self.home_button.grid(row=8, column=0)

    def vid_thread(self, path, stud_name, sm):
        if os.path.isfile(path):
            extension = os.path.splitext(path)[1]
            if (extension.lower() == ".mp4" or extension.lower() == ".avi" or extension.lower() == ".mov"
                    or extension.lower() == ".wmv" or extension.lower() == ".mkv"):
                connection = sqlite3.connect("real.db")
                cursor = connection.cursor()
                cursor.execute("SELECT * FROM all_students WHERE name = ?", (self.stud_entry.get(),))
                if cursor.fetchall():
                    self.name = self.stud_entry.get()
                    cursor.execute("UPDATE all_students SET videos = videos + 1 WHERE name = ?",
                                   (self.stud_entry.get(),))
                    connection.commit()
                    connection.close()
                    t1 = threading.Thread(target=vid_analysis, args=(path, self.df, stud_name, sm))
                    t1.start()
                else:
                    messagebox.showerror("Error", "Student Not Found")
            else:
                messagebox.showerror("Error", "Incorrect File Type")
        else:
            messagebox.showerror("Error", "File Does Not Exist")

    def pic_thread(self, path, stud_name, sm):
        if os.path.isfile(path):
            extension = os.path.splitext(path)[1]
            if extension.lower() == ".jpg" or extension.lower() == ".jpeg" or extension.lower() == ".png":
                connection = sqlite3.connect("real.db")
                cursor = connection.cursor()
                cursor.execute("SELECT * FROM all_students WHERE name = ?", (self.stud_entry.get(),))
                if cursor.fetchall():
                    self.name = self.stud_entry.get()
                    cursor.execute("UPDATE all_students SET photos = photos + 1 WHERE name = ?",
                                   (self.stud_entry.get(),))
                    connection.commit()
                    connection.close()
                else:
                    messagebox.showerror("Error", "Student Not Found")
                t2 = threading.Thread(target=pic_analysis, args=(path, self.df, stud_name, sm))
                t2.start()
            else:
                messagebox.showerror("Error", "Incorrect File Type")
        else:
            messagebox.showerror("Error", "File Does Not Exist")

    def home_switch(self):
        self.root.destroy()
        home_screen(self.df, self.sm)


class EditStudentsScreen:
    def __init__(self, root, df, sm):
        self.sm = sm
        self.df = df
        self.root = root
        self.root.title = "Edit Students"
        self.root.geometry("2560x1440")

        self.add_label = tk.Label(root, text="Enter Student Name to Add")
        self.add_label.grid(row=0, column=0)

        self.del_label = tk.Label(root, text="Enter Student Name to Delete")
        self.del_label.grid(row=1, column=0)

        self.pin_label = tk.Label(root, text="Enter Admin Pin")
        self.pin_label.grid(row=2, column=0)

        self.stud_entry = tk.Entry(root)
        self.stud_entry.grid(row=0, column=1)

        self.del_entry = tk.Entry(root)
        self.del_entry.grid(row=1, column=1)

        self.pin_entry = tk.Entry(root, show='*')
        self.pin_entry.grid(row=2, column=1)

        self.submit_button = tk.Button(root, text="Add", command=self.add_student)
        self.submit_button.grid(row=0, column=2)

        self.del_button = tk.Button(root, text="Delete", command=self.del_student)
        self.del_button.grid(row=1, column=2)

        self.home_button_two = tk.Button(root, text="Home", command=self.home_switch)
        self.home_button_two.grid(row=8, column=0)

    def add_student(self):
        if self.stud_entry.get().isalpha():
            if self.pin_entry.get() == "password":
                name = self.stud_entry.get()
                with open(self.df, 'a') as f:
                    text = "\nStudent: " + name
                    f.write(text)
                    connection = sqlite3.connect("real.db")
                    cursor = connection.cursor()
                    cursor.execute("INSERT INTO all_students (name, videos, photos) VALUES (?,?,?)", (name, 0, 0))
                    cursor.execute("SELECT * FROM all_students ORDER BY name")
                    cursor.close()
                    connection.commit()
                    connection.close()

                    messagebox.showinfo("Success", "Student Added")
            else:
                messagebox.showerror("Error", "Incorrect Password or Invalid Name")
        else:
            messagebox.showerror("Error", "Incorrect Password or Invalid Name")

    def del_student(self):
        if self.del_entry.get().isalpha():
            if self.pin_entry.get() == "password":
                with open(self.df, 'r') as f:
                    target = "Student: " + self.del_entry.get()
                    length = len(target)
                    content = f.read()
                    index = content.find(target)
                    if index != -1:
                        next_target = "Student: "
                        next_index = content.find(next_target, index + length)  # starting pos for searching
                        if next_index != -1:
                            new_content = content[:index] + content[next_index:]
                        else:
                            new_content = content[:index]
                        with open(self.df, 'w') as file:
                            file.write(new_content)
                    else:
                        messagebox.showerror("Error", "Student Not Found. Make Sure Spelling is Correct.")

                connection = sqlite3.connect("real.db")
                cursor = connection.cursor()
                cursor.execute("DELETE FROM all_students WHERE name=?", (self.del_entry.get(),))
                connection.commit()
                connection.close()
                messagebox.showinfo("Success", "Student Deleted")
            else:
                messagebox.showerror("Error", "Incorrect Password or Invalid Name")
        else:
            messagebox.showerror("Error", "Incorrect Password or Invalid Name")

    def home_switch(self):
        self.root.destroy()
        home_screen(self.df, self.sm)


def start(df, sm):
    root = tk.Tk()
    LoginScreen(root, df, sm)
    root.mainloop()


def home_screen(df, sm):
    root = tk.Tk()
    HomeScreen(root, df, sm)
    root.mainloop()


def stud_screen(df, sm):
    root = tk.Tk()
    EditStudentsScreen(root, df, sm)
    root.mainloop()


def funct_screen(df, sm):
    root = tk.Tk()
    FunctionScreen(root, df, sm)
    root.mainloop()


if __name__ == "__main__":
    start(data_file, system_message)
