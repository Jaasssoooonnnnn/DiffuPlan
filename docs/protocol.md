# Agent protocol

Action Mode emits a plan in the first segment, followed by reasoning and one terminal action. Terminal actions are search, update, replan, and answer. Information blocks come from the retriever.

Use newline-separated blocks. Update and replan payloads contain comma-separated markers without spaces.

    <plan>
    #Q_1: Which person wrote the novel?
    #Q_2: In which city was #A_1 born?
    </plan>
    <think>
    I will identify the author first.
    </think>
    <search>
    author of the novel
    </search>

An update such as the following hands control to Edit Mode:

    <think>
    The retrieved passage identifies the author.
    </think>
    <update>
    #A_1
    </update>

The controller masks the selected reference in the existing plan. The editor sees the full interaction history, including the evidence after that plan, and reconstructs the replacement. Commit propagates the value to identical unresolved references.

A single replan marker selects the suffix starting at that subquestion; multiple markers select the listed existing step spans. Unselected plan content and the surrounding retrieval history are retained.

The reserved token <|reserved_token_6|> is used as the internal expansion token. It is permitted only in Edit Mode. Action sampling bans mask and expansion tokens. EPM supervises alternative valid allocations of replacement tokens to edit masks.

Action SFT supervises the next complete assistant segment from its actual history. Edit SFT supervises owned masks in the full canvas. Retrieval observations are conditioning context. The released RL configuration applies the trajectory objective and frozen SFT anchor to Action-Mode tokens; Edit Mode shares the updated model parameters.
