from output import company_row


def run_icp(icp, tools, model):
    observations = []
    while True:
        choice = model.decide(icp, observations)
        if choice["action"] == "finish":
            return [row for record in observations if (row := company_row(record, icp))]
        observations.extend(tools.call(choice["action"], choice["arguments"]))
