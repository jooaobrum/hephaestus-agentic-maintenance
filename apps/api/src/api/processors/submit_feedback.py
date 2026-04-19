from langsmith import Client

client = Client()


def submit_feedback(
    trace_id: str,
    feedback_value: int = None,
    feedback_text: str = None,
    feedback_source_type: str = "api",
):
    """
    Submits feedback to LangSmith.
    """
    try:
        # Submit the numerical score (thumbs up/down)
        if feedback_value is not None:
            client.create_feedback(
                run_id=trace_id,
                score=feedback_value,
                key="thumbs",
                feedback_source_type=feedback_source_type,
            )

        # Submit the text comment if provided
        if feedback_text and len(feedback_text.strip()) > 0:
            client.create_feedback(
                run_id=trace_id,
                value=feedback_text,
                key="comment",
                feedback_source_type=feedback_source_type,
            )

        return "success", "Feedback submitted successfully"
    except Exception as e:
        return "failed", str(e)
