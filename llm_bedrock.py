import boto3
import llm
import os
import pathlib

MODELS = (
    # model_id, alias, supports_attachments
    ("us.amazon.nova-micro-v1:0", "nova-micro", False),
    ("us.amazon.nova-lite-v1:0", "nova-lite", True),
    ("us.amazon.nova-pro-v1:0", "nova-pro", True),
)
AWS_REGION = os.environ["AWS_REGION"] if "AWS_REGION" in os.environ else "us-west-2"


@llm.hookimpl
def register_models(register):
    for model_id, alias, supports_attachments in MODELS:
        register(BedrockModel(model_id, supports_attachments), aliases=[alias])


FORMAT_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/webp": "webp",
    "image/gif": "gif",
    "application/pdf": "pdf",
    "video/quicktime": "mov",
    "video/x-matroska": "mkv",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/x-flv": "flv",
    "video/mpeg": "mpeg",
    "video/x-ms-wmv": "wmv",
}


class BedrockModel(llm.Model):
    needs_key = "bedrock"
    can_stream = True

    def __init__(self, model_id, supports_attachments):
        self.model_id = model_id
        if supports_attachments:
            self.attachment_types = {
                "image/png",
                "image/jpeg",
                "image/webp",
                "image/gif",
                "application/pdf",
                "video/quicktime",
                "video/x-matroska",
                "video/mp4",
                "video/webm",
                "video/x-flv",
                "video/mpeg",
                "video/mpeg",
                "video/x-ms-wmv",
            }

    def _user_message(self, prompt):
        content = []
        for attachment in prompt.attachments:
            if attachment.type.startswith("image/"):
                content.append(
                    {
                        "image": {
                            "format": FORMAT_TYPES[attachment.type],
                            "source": {"bytes": attachment.content_bytes()},
                        }
                    }
                )
            elif attachment.type.startswith("video/"):
                content.append(
                    {
                        "video": {
                            "format": FORMAT_TYPES[attachment.type],
                            "source": {"bytes": attachment.content_bytes()},
                        }
                    }
                )
            elif attachment.type == "application/pdf":
                # Name is required
                name = ""
                if attachment.path:
                    name = pathlib.Path(attachment.path).stem
                    # Remove any non-alphanumeric characters
                    name = "".join(c for c in name if c.isalnum())
                if not name:
                    name = "attachment"
                content.append(
                    {
                        "document": {
                            "name": name,
                            "format": "pdf",
                            "source": {"bytes": attachment.content_bytes()},
                        }
                    }
                )
        content.append({"text": prompt.prompt})
        return {"role": "user", "content": content}

    def execute(self, prompt, stream, response, conversation):
        try:
            key = self.get_key()
            # Check if key contains a colon - if so, treat as access_key:secret_key
            if ":" in key:
                access_key, _, secret_key = key.partition(":")
                session = boto3.Session(
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                )
            else:
                # Otherwise treat as profile name
                session = boto3.Session(profile_name=key)
        except llm.errors.NeedsKeyException:
            # Try fallback on boto3.Session() default
            session = boto3.Session()
            if not session.get_credentials():
                raise llm.errors.NeedsKeyException("No AWS credentials found")

        # Create Bedrock client
        bedrock = session.client("bedrock-runtime", region_name=AWS_REGION)
        messages = []
        system = prompt.system
        if conversation:
            for turn in conversation.responses:
                if not system and turn.prompt.system:
                    system = turn.prompt.system
                messages.append(self._user_message(turn.prompt))
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"text": turn.text_or_raise()},
                        ],
                    }
                )
        messages.append(self._user_message(prompt))
        params = {"messages": messages, "modelId": self.model_id}
        if system:
            params["system"] = [{"text": system}]
        chunks = []
        usage = {}
        if stream:
            bedrock_response = bedrock.converse_stream(**params)
            for event in bedrock_response["stream"]:
                chunks.append(event)
                ((event_type, event_content),) = event.items()
                if event_type == "contentBlockDelta":
                    yield event_content["delta"]["text"]
                if "metadata" in event and "usage" in event["metadata"]:
                    usage = event["metadata"]["usage"]
        else:
            bedrock_response = bedrock.converse(**params)
            chunk = bedrock_response["output"]["message"]["content"][-1]["text"]
            yield chunk
            usage = bedrock_response["usage"]

        if usage:
            response.set_usage(input=usage["inputTokens"], output=usage["outputTokens"])

    def __str__(self):
        return "Bedrock: {}".format(self.model_id)
