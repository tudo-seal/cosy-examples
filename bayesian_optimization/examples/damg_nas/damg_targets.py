from cosy.core.types import Constructor, Literal

from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository

# Note that the literal value in Constructor("epochs", Literal(10000)) is current not allowed to be None!
target_len_6 = Constructor("Learner", Constructor("DAG",
                                                          Constructor("input", Literal(1))
                                                          & Constructor("output", Literal(1))
                                                          & Constructor("structure", Literal(
                                                              (
                                                                  None,  # left, split, right
                                                                  None,  # left, gate, right
                                                                  None,  # left, gate, right
                                                                  None,  # left_out, -gate, right
                                                                  None,  # left_out, 1-gate, right
                                                                  None,  # left_out, right_out
                                                              )
                                                          )))
                                   & Constructor("Loss", Constructor("type", Literal(None)))
                                   & Constructor("Optimizer", Constructor("type", Literal(None)))
                                   & Constructor("epochs", Literal(2000))
                                   )

target_len_5 = Constructor("Learner", Constructor("DAG",
                                                          Constructor("input", Literal(1))
                                                          & Constructor("output", Literal(1))
                                                          & Constructor("structure", Literal(
                                                              (
                                                                  None,  # left, split, right
                                                                  None,  # left, gate, right
                                                                  None,  # left_out, -gate, right
                                                                  None,  # left_out, 1-gate, right
                                                                  None,  # left_out, right_out
                                                              )
                                                          )))
                                   & Constructor("Loss", Constructor("type", Literal(None)))
                                   & Constructor("Optimizer", Constructor("type", Literal(None)))
                                   & Constructor("epochs", Literal(2000))
                                   )

target_len_4 = Constructor("Learner", Constructor("DAG",
                                                          Constructor("input", Literal(1))
                                                          & Constructor("output", Literal(1))
                                                          & Constructor("structure", Literal(
                                                              (
                                                                  None,  # left, split, right
                                                                  None,  # left, gate, right
                                                                  None,  # left_out, -gate, right
                                                                  None,  # left_out, 1-gate, right
                                                              )
                                                          )))
                                   & Constructor("Loss", Constructor("type", Literal(None)))
                                   & Constructor("Optimizer", Constructor("type", Literal(None)))
                                   & Constructor("epochs", Literal(2000))
                                   )


target_len_3 = Constructor("Learner", Constructor("DAG",
                                                          Constructor("input", Literal(1))
                                                          & Constructor("output", Literal(1))
                                                          & Constructor("structure", Literal(
                                                              (
                                                                  None,  # left, gate, right
                                                                  None,  # left_out, -gate, right
                                                                  None,  # left_out, 1-gate, right
                                                              )
                                                          )))
                                   & Constructor("Loss", Constructor("type", Literal(None)))
                                   & Constructor("Optimizer", Constructor("type", Literal(None)))
                                   & Constructor("epochs", Literal(2000))
                                   )

target_len_2 = Constructor("Learner", Constructor("DAG",
                                                          Constructor("input", Literal(1))
                                                          & Constructor("output", Literal(1))
                                                          & Constructor("structure", Literal(
                                                              (
                                                                  None,  # left, gate, right
                                                                  None,  # left_out, -gate, right
                                                              )
                                                          )))
                                   & Constructor("Loss", Constructor("type", Literal(None)))
                                   & Constructor("Optimizer", Constructor("type", Literal(None)))
                                   & Constructor("epochs", Literal(2000))
                                   )

target_len_3_refined_1 = Constructor("Learner", Constructor("DAG",
                                                            Constructor("input", Literal(1))
                                                            & Constructor("output", Literal(1))
                                                            & Constructor("structure", Literal(
                                                      (
                                                          (
                                                              None,
                                                          ),
                                                          None,
                                                          (
                                                              None,
                                                          ),
                                                      )
                                                  )))
                                     & Constructor("Loss", Constructor("type", Literal(None)))
                                     & Constructor("Optimizer", Constructor("type", Literal(None)))
                                     & Constructor("epochs", Literal(2000))
                                     )

target_len_3_refined_2 = Constructor("Learner", Constructor("DAG",
                                                            Constructor("input", Literal(1))
                                                            & Constructor("output", Literal(1))
                                                            & Constructor("structure", Literal(
                                                      (
                                                          (
                                                              None,
                                                          ),
                                                          (
                                                              None,
                                                              None,
                                                          ),
                                                          (
                                                              None,
                                                          ),
                                                      )
                                                  )))
                                     & Constructor("Loss", Constructor("type", Literal(None)))
                                     & Constructor("Optimizer", Constructor("type", Literal(None)))
                                     & Constructor("epochs", Literal(2000))
                                     )

target_len_3_refined_3 = Constructor("Learner", Constructor("DAG",
                                                            Constructor("input", Literal(1))
                                                            & Constructor("output", Literal(1))
                                                            & Constructor("structure", Literal(
                                                      (
                                                          (
                                                              None,
                                                          ),
                                                          (
                                                              None,
                                                              None,
                                                              None,
                                                          ),
                                                          (
                                                              None,
                                                          ),
                                                      )
                                                  )))
                                     & Constructor("Loss", Constructor("type", Literal(None)))
                                     & Constructor("Optimizer", Constructor("type", Literal(None)))
                                     & Constructor("epochs", Literal(2000))
                                     )

target_len_5_refined_1 = Constructor("Learner", Constructor("DAG",
                                                            Constructor("input", Literal(1))
                                                            & Constructor("output", Literal(1))
                                                            & Constructor("structure", Literal(
                                                      (
                                                          (
                                                              None,
                                                          ),
                                                          None,
                                                          None,
                                                          None,
                                                          (
                                                              None,
                                                          ),
                                                      )
                                                  )))
                                     & Constructor("Loss", Constructor("type", Literal(None)))
                                     & Constructor("Optimizer", Constructor("type", Literal(None)))
                                     & Constructor("epochs", Literal(2000))
                                     )

target_len_5_refined_2 = Constructor("Learner", Constructor("DAG",
                                                            Constructor("input", Literal(1))
                                                            & Constructor("output", Literal(1))
                                                            & Constructor("structure", Literal(
                                                      (
                                                          (
                                                              None,
                                                          ),
                                                          (
                                                              None,
                                                              None,
                                                              None,
                                                          ),
                                                          None,
                                                          (
                                                              None,
                                                              None,
                                                          ),
                                                          (
                                                              None,
                                                          ),
                                                      )
                                                  )))
                                     & Constructor("Loss", Constructor("type", Literal(None)))
                                     & Constructor("Optimizer", Constructor("type", Literal(None)))
                                     & Constructor("epochs", Literal(2000))
                                     )

target_len_5_refined_3 = Constructor("Learner", Constructor("DAG",
                                                            Constructor("input", Literal(1))
                                                            & Constructor("output", Literal(1))
                                                            & Constructor("structure", Literal(
                                                      (
                                                          (
                                                              None,
                                                          ),
                                                          (
                                                              None,
                                                              None,
                                                              None,
                                                          ),
                                                          (
                                                              None,
                                                              None,
                                                              None,
                                                          ),
                                                          (
                                                              None,
                                                              None,
                                                          ),
                                                          (
                                                              None,
                                                          ),
                                                      )
                                                  )))
                                     & Constructor("Loss", Constructor("type", Literal(None)))
                                     & Constructor("Optimizer", Constructor("type", Literal(None)))
                                     & Constructor("epochs", Literal(2000))
                                     )

target_len_4_refined_1 = Constructor("Learner", Constructor("DAG",
                                                          Constructor("input", Literal(1))
                                                          & Constructor("output", Literal(1))
                                                          & Constructor("structure", Literal(
                                                              (
                                                                  (
                                                                      None,
                                                                  ),
                                                                  None,
                                                                  None,
                                                                  (
                                                                      None,
                                                                  ),
                                                              )
                                                          )))
                                   & Constructor("Loss", Constructor("type", Literal(None)))
                                   & Constructor("Optimizer", Constructor("type", Literal(None)))
                                   & Constructor("epochs", Literal(2000))
                                   )

def target_to_name(target):
    if target == target_len_2:
        return "target_len_2"
    if target == target_len_3:
        return "target_len_3"
    if target == target_len_4:
        return "target_len_4"
    if target == target_len_5:
        return "target_len_5"
    if target == target_len_6:
        return "target_len_6"
    if target == target_len_3_refined_1:
        return "target_len_3_refined_1"
    if target == target_len_3_refined_2:
        return "target_len_3_refined_2"
    if target == target_len_3_refined_3:
        return "target_len_3_refined_3"
    if target == target_len_4_refined_1:
        return "target_len_4_refined_1"
    if target == target_len_5_refined_1:
        return "target_len_5_refined_1"
    if target == target_len_5_refined_2:
        return "target_len_5_refined_2"
    if target == target_len_5_refined_3:
        return "target_len_5_refined_3"
    return "unknown"
