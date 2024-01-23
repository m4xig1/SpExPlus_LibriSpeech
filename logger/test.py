if __name__ == "__main__":
    from logger import *
    from visualize import *
    from config import *

    v = WandbVisualizer(config_wdb)
    v.log_text(__name__, "test text")
    v.log_scalar("two", 2)
    v.new_step(2)
    # v.log_audio(__name__, )

    # start_log(config_log)
    # log = logging.Logger(__name__)
    # log.info("test msg")

