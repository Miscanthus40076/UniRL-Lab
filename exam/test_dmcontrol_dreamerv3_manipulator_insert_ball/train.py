from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import BaseExamTrainApp


class ExamTrain(BaseExamTrainApp):
    def build_trainer(self, config: dict):
        return self.build_default_trainer(config)


def main():
    return ExamTrain.from_source_file(__file__).run()


if __name__ == "__main__":
    main()
