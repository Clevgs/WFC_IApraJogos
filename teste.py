import wfc.wfc_control as wfc_control

wfc_control.execute_wfc(
                        filename= "Mapa_para_WFC2",
                        tile_size=1,
                        pattern_width=2,
                        rotations=8,
                        output_size=[40,40],
                        ground=-1,
                        attempt_limit=5,
                        output_periodic=True,
                        input_periodic=True,
                        loc_heuristic="entropy",
                        choice_heuristic="weighted ",
                        backtracking=True,
                        global_constraint="allpatterns",
                        log_filename="tes",
                        log_stats_to_output=None,
                        visualize=True,
                        logging=True,
                    )