from llm.recommender import Recommender
from llm.interaction import Interaction
from robot.controller import Controller
from robot.detic import DeticPredictor
from llm.find_books import BookFinder
from enum import Enum
from threading import Thread
import rospy

class State(Enum):
    WAIT = 0
    RECOMMEND = 1
    FOR = 2
    BACK = 3
    PICK = 4

ORIGIN = "instruction_point"

OUTER_BOOKS = [
    "ゴールデンカムイ",
    "無職転生",
    "pythonで学ぶアルゴリズムとデータ構造",
    "ロシア語でボソッとデレるアーリャさん",
    "プラチナコード",
    "図書館戦争"
]

class Flow():
    def __init__(self):
        rospy.init_node("my_flow", anonymous=True)

        self.controller = Controller(location_file_path="/root/HSR/catkin_ws/src/gpsr/scripts/spotting_data/kawa5.json")
        self.recommender = Recommender("books.db", verbose=True)
        self.recommender.insert_books(self.controller.books, place=0)
        self.recommender.insert_books(OUTER_BOOKS, place=1) # TODO: ここに本の内容を入れる
        self.detector = DeticPredictor(vocabulary="lvis", titles=OUTER_BOOKS)
        self.interaction = Interaction()

        self.state = State.WAIT
        self.controller.speak("全ての初期化が完了しました．")

    def process(self):
        # Start the Detection Thread
        Thread(target=self.detector.process).start()

        while not rospy.is_shutdown():
            try:
                if self.state == State.WAIT:
                    self.detector.should_detect = True
                    is_listen_success, sentence = self.controller.listen()
                    print(is_listen_success, sentence)
                    print(self.detector.book_name)
                    

                    if not is_listen_success:
                        continue

                    else:
                        print("flag")

                        response = self.interaction.interact(sentence)
                        print(response)
                        if response == "<REQ>":
                            self.state = State.RECOMMEND
                            continue
                        else:
                            self.controller.speak(response)
                            continue

                elif self.state == State.RECOMMEND:
                    self.controller.speak("どんな本が読みたいですか？")
                    if self.detector.book_name is not None:
                        top_book = self.recommender.get_recommendations_from_title(self.detector.book_name)
                    else:
                        is_listen_success, query = self.controller.listen()
                        if not is_listen_success:
                            self.controller.speak("聞き取りがうまくいかなかったようです．もう一度お願いします．")
                            continue

                        top_book = self.recommender.get_recommendations_from_query(query)
                    self.detector.should_detect = False
                    self.controller.speak(f"わかりました．好みの本を探します．しばらく時間がかかります")
                    self.controller.speak(f"多分{top_book}がおすすめです．")
                    self.state = State.FOR
                    self.controller.speak("本を取りに行きます．")
                    self.controller.move_to_with_name(top_book)
                    self.state = State.PICK
                     
                elif self.state == State.PICK:
                    self.controller.pick()
                    self.controller.speak("本を取りました．元の場所に戻ります．")
                    self.state = State.BACK
                    self.controller.move_to_with_name(ORIGIN)
                    self.controller.speak("本を離します")
                    self.controller.release()
                    self.state = State.WAIT

            except KeyboardInterrupt:
                self.controller.speak("プログラムを終了します．")
                break

            except:
                self.controller.speak("エラーが発生しました．初期状態に戻ります．")
                self.controller.neutral()
                self.state = State.BACK
                self.controller.move_to_with_name(ORIGIN)
                self.state = State.WAIT
                raise ValueError("Invalid State")
            

if __name__ == "__main__":
    flow = Flow()
    flow.process()
    rospy.spin()