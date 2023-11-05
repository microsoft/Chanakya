mab_configs = {
    "frcnn_esp_256" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 720), (2000, 640), (2000, 560), (2000, 480), (2000, 360) ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.999,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [1, 1] # 300, 640
        }
    },
    "frcnn_esp_imagenet_vid_256" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 44
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 600), (2000, 480), (2000, 420), (2000, 360), (2000, 300), (2000, 240) ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.999,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [1, 1] # 300, 640
        }
    },
    "frcnn_esp_imagenet_vid_256_r2" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 44
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 600), (2000, 480), (2000, 420), (2000, 360), (2000, 300), (2000, 240) ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.999,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [1, 1] # 300, 640
        }
    },
    "frcnn_ucb_256" : {
        "states" : {
            "keys" : { "curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 720), (2000, 640), (2000, 560), (2000, 480), (2000, 360) ] ]
        ],
        "rl" : {
            "algo" : "ucb",
            "c" : 1.5,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [1, 1] # 300, 640
        }
    },
    "frcnn_esp_256_v2" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 720), (2000, 640), (2000, 560), (2000, 480), (2000, 360) ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.999,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [1, 1] # 300, 640
        }
    },
    "frcnn_esp_256_fixed_adv_large_action" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 720), (2000, 640), (2000, 600), (2000, 560), (2000, 480), (2000, 360) ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.999,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [1, 1] # 300, 640
        }
    },
    "frcnn_esp_256_changed_scale" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 750), (2000, 675), (2000, 600), (2000, 525), (2000, 450) ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.999,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [1, 1] # 300, 675
        }
    },
    "frcnn_esp_256_fixed_adv" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 720), (2000, 640), (2000, 560), (2000, 480), (2000, 360) ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.999,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [1, 1] # 300, 640
        }
    },
    "frcnn_esp_256_fixed_adv_480_300" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 720), (2000, 640), (2000, 560), (2000, 480), (2000, 360) ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.999,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [1, 1] # 300, 640
        }
    },
    "frcnn_esp_256_changed_scale_fixed_adv_480_300" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 750), (2000, 675), (2000, 600), (2000, 525), (2000, 450) ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.999,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [1, 1] # 300, 675
        }
    },
    "frcnn_esp_256_all_metrics_fixed_adv_480_300" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop", "switch_score",  "area_info"},
            "state_size" : 22
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 720), (2000, 640), (2000, 560), (2000, 480), (2000, 360) ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.999,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [1, 1] # 300, 640
        }
    },
    "frcnn_esp_256_tracking_fixed_adv_480_300" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop", "switch_score",  "area_info"},
            "state_size" : 22
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 720), (2000, 640), (2000, 560), (2000, 480), (2000, 360) ] ],
            [ "tra_scale" , [ (2000, 720), (2000, 640), (2000, 560), (2000, 480), (2000, 360) ] ],
            [ "tra_stride", [ 3, 5, 10, 15 ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.999,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [1, 1, 1, 0] # 300, 640
        }
    },
    "frcnn_esp_256_fixed_adv_switching_480_300" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop", "switch_score",  "area_info"},
            "state_size" : 22
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 720), (2000, 640), (2000, 560), (2000, 480), (2000, 360) ] ],
            [ "switch" , [ "faster_rcnn", "fcos", "yolov3" ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.9995,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [1, 1, 0] # 300, 640
        }
    },
    "frcnn_esp_imagenet_vid_256_r2_switching" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop", "switch_score",  "area_info"},
            "state_size" : 44
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [  (2000, 600), (2000, 480), (2000, 420), (2000, 360), (2000, 300), (2000, 240)  ] ],
            [ "switch" , [ "faster_rcnn", "fcos", "yolov3" ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.9995,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [1, 1, 0] # 300, 640
        }
    },
    "frcnn_esp_256_fixed_adv_switching_explorer_480_300" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop", "switch_score",  "area_info"},
            "state_size" : 22
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 720), (2000, 640), (2000, 560), (2000, 480), (2000, 360) ] ],
            [ "switch" , [ "faster_rcnn", "fcos", "yolov3" ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.9997,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [1, 1, 0] # 300, 640
        }
    },
    "frcnn_esp_256_fixed_adv_only_switching_480_300" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop", "switch_score",  "area_info"},
            "state_size" : 22
        },
        "actions" : [
            [ "prop" , [300] ],
            [ "det_scale" , [ (2000, 640) ] ],
            [ "switch" , [ "faster_rcnn", "fcos", "yolov3" ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.999,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [0, 0, 0] # 300, 640
        }
    },
    "frcnn_esp_256_fixed_adv_360_1000" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 720), (2000, 640), (2000, 560), (2000, 480), (2000, 360) ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.999,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [1, 1] # 300, 640
        }
    },
    "frcnn_ucb_256_changed_scale" : {
        "states" : {
            "keys" : { "curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 750), (2000, 675), (2000, 600), (2000, 525), (2000, 450) ] ]
        ],
        "rl" : {
            "algo" : "ucb",
            "c" : 1.5,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [1, 1] # 300, 675
        }
    },
    "frcnn_esp_256_fixed_adv_480_vs_640" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop" , [300] ],
            [ "det_scale" , [  (2000, 640) ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.999,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [0, 0] # 300, 640
        }
    },
    "frcnn_esp_256_fixed_adv_480_vs_640_best_seqs" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop" , [300] ],
            [ "det_scale" , [  (2000, 640) ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.999,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [0, 0] # 300, 640
        }
    },
    "frcnn_esp_256_fixed_adv_480_vs_640_worst_seqs" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop" , [300] ],
            [ "det_scale" , [  (2000, 640) ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.999,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [0, 0] # 300, 640
        }
    },
    "frcnn_esp_256_fixed_prop" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop" , [500] ],
            [ "det_scale" , [ (2000, 720), (2000, 640), (2000, 560), (2000, 480), (2000, 360) ] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 0.999,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [0, 1] # 500, 640
        }
    },
    "frcnn_ucb_64_fixed_prop" : {
        "states" : {
            "keys" : { "curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop" , [500] ],
            [ "det_scale" , [ (2000, 720), (2000, 640), (2000, 560), (2000, 480), (2000, 360) ] ]
        ],
        "rl" : {
            "algo" : "ucb",
            "c" : 1.75,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 64,
                "batch_size" : 32
            }
        }
    },
    "frcnn_ucb_64_fixed_prop_coco" : {
        "states" : {
            "keys" : { "curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop" , [300] ],
            [ "det_scale" , [ (2000, 720), (2000, 640), (2000, 560), (2000, 480), (2000, 360) ] ]
        ],
        "rl" : {
            "algo" : "ucb",
            "c" : 1.75,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 64,
                "batch_size" : 32
            }
        }
    },
    "frcnn_ucb_1d" : {
        "states" : {
            "keys" : { "curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop_det_scale" , [ 
                [100, (2000, 720)], [300, (2000, 720)], [500, (2000, 720)], 
                [100, (2000, 640)], [300, (2000, 640)], [500, (2000, 640)], [1000, (2000, 640)], 
                [100, (2000, 480)], [300, (2000, 480)], 
                [100, (2000, 360)] ] 
            ]
        ],
        "rl" : {
            "algo" : "ucb",
            "c" : 1.75,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            }
        }
    },
    "frcnn_esp_256_mixed" : {
        "states" : {
            "keys" : {"curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop" },
            "state_size" : 16
        },
        "actions" : [
            [ "prop" , [300] ],
            [ "det_scale" , [ (2000, 720), (2000, 640)] ]
        ],
        "rl" : {
            "algo" : "esp",
            "epsilon" : 1,
            "min_epsilon" : 0.15,
            "epsilon_decay" : 1,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 32
            },
            "init_action" : [0, 1] # 300, 640
        }
    },
    "frcnn_ucb_256_complexity_3_models" : {
        "states" : {
            "keys" : { "curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop", "switch" },
            "state_size" : 19
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 720), (2000, 640), (2000, 560), (2000, 480), (2000, 360) ] ],
            [ "switch", ["yolov3", "fcos", "faster_rcnn"]]
        ],
        "rl" : {
            "algo" : "ucb",
            "c" : 1.75,
            "mlp" : {
                "num_layers" : 6,
                "hidden_layer_size" : 256,
                "batch_size" : 16
            }
        }
    },
    "frcnn_ucb_256_complexity_2_models_two_stage" : {
        "states" : {
            "keys" : { "curr_scale", "ada_scale", "conf_info", "class_info", "tb_lr_crop", "switch" },
            "state_size" : 19
        },
        "actions" : [
            [ "prop" , [100, 300, 500, 1000] ],
            [ "det_scale" , [ (2000, 720), (2000, 640), (2000, 560), (2000, 480), (2000, 360) ] ],
            [ "switch", ["cascade_rcnn", "faster_rcnn"]]
        ],
        "rl" : {
            "algo" : "ucb",
            "c" : 1.75,
            "mlp" : {
                "num_layers" : 4,
                "hidden_layer_size" : 256,
                "batch_size" : 16
            }
        }
    }
}

# 4, 256