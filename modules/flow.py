from llm.recommender import Recommender
from llm.interaction import Interaction
from robot.controller import Controller
from enum import Enum

class State(Enum):
    WAIT = 0
    RECOMMEND = 1
    FOR = 2
    BACK = 3
    PICK = 4


class Flow():
    def __init__(self):

        self.controller = Controller(location_file_path="locations.json")
        self.recommender = Recommender("books.db", self.controller.books, verbose=False)
        self.interaction = Interaction()

        self.state = State.WAIT
        self.controller.speak("全ての初期化が完了しました．")

    def process(self):
        while True:
            try:
                if self.state == State.WAIT:
                    is_listen_success, sentence = self.controller.listen()

                    if not is_listen_success:
                        continue

                    elif sentence == "<REQ>":
                        self.state = State.RECOMMEND
                        continue

                    else:
                        response = self.interaction.interact(sentence)
                        self.controller.speak(response)
                        continue

                elif self.state == State.RECOMMEND:
                    self.controller.speak("どんな本が読みたいですか？")
                    is_listen_success, query = self.controller.listen()

                    if not is_listen_success:
                        self.controller.speak("聞き取りがうまくいかなかったようです．もう一度お願いします．")
                        continue

                    self.controller.speak(f"わかりました．{query}の特徴を持つ本を探します．")
                    top_book = self.recommender.get_recommendations(query)
                    self.controller.speak(f"多分{top_book}がおすすめです．")
                    self.state = State.FOR
                    self.controller.speak("本を取りに行きます．")
                    self.controller.move_to_with_name(top_book)
                    self.state = State.PICK
                     
                elif self.state == State.PICK:
                    self.controller.pick()
                    self.controller.speak("本を取りました．元の場所に戻ります．")
                    self.controller.move_to_with_name("initial")
                    self.state = State.BACK

            except KeyboardInterrupt:
                self.controller.speak("プログラムを終了します．")
                break

            except:
                self.controller.speak("エラーが発生しました．初期状態に戻ります．")
                self.controller.neutral()
                self.state = State.BACK
                self.controller.move_to_with_name("initial")
                self.state = State.WAIT
                raise ValueError("Invalid State")
            

if __name__ == "__main__":
    flow = Flow()
    flow.process()