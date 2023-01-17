from utils.Option import param
# from Runner.trainer import train
from Runner.official_trainer import train
from Runner.official_tester import test


class driver(param):
    def __init__(self):
        super(driver, self).__init__()

    def run_train(self):
        tr = train()
        tr.run()

    def run_test(self):
        te = test()
        te.run()

    def __call__(self, *args, **kwargs):
        if self.run_type == 0:
            return self.run_train()

        if self.run_type == 1:
            return self.run_test()


if __name__ == "__main__":
    driver()()