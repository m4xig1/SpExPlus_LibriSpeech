import logging

config_trainer = {
    "logger": {"level": logging.INFO},
    "optimizer" : {
        "lr" : 0.01,
        "weight_decay" : 1e-5
    },
    "lrScheduler" : { # Reduce on plateau
        "mode" : "min", 
        "factor" : 0.5,
        "patience" : 0,
        "min_lr" : 0, 
        "verbose" : True
    },
      "nCheckpoints": 5}
