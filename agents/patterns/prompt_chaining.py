import os
from openai import AzureOpenAI
from pydantic import BaseModel, Field
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

model = "gpt-4o-mini-2"
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
api_version = os.getenv("API_VERSION")

if not azure_endpoint:
    logger.critical("Azure endpoint is not set. Please check your .env file.")
    raise ValueError(
        "The value of azure endpoint is None. Please check your .env file."
    )

client = AzureOpenAI(
    api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint
)

"""
Pydantic Models

These models define the structured outputs expected from the LLM
for various stages of email processing, such as spam filtering,
content cleaning, and summarization.
"""


class EmailInput(BaseModel):
    subject: str
    sender: str
    recipient: str
    body: str


class FilteredEmail(BaseModel):
    is_spam: bool = Field(description="determine if the email is spam or not")
    confidence_score: float = Field(
        ge=0, le=1, description="how confident are you about the email is spam or not"
    )


class CleanedEmailOutput(BaseModel):
    clean_output: str = Field(
        description="clean and the meaningful content (remove signatures, disclaimers, quoted replies, etc.)"
    )


class EmailSummaryOutput(BaseModel):
    summary: str = Field(description="concise summary of an email")


def filter_mail(mail: EmailInput) -> FilteredEmail:
    """
    Filters an email to determine if it is spam.

    Args:
        mail (EmailInput): The email to be filtered.

    Returns:
        FilteredEmail: The result indicating if the email is spam and the confidence score.

    Raises:
        Exception: If the LLM call fails or returns no result.
    """
    logger.info(
        f"Starting spam filtering for email with subject: '{mail.subject}' from '{mail.sender}'"
    )
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that filters the email provided.",
                },
                {"role": "user", "content": str(mail)},
            ],
            response_format=FilteredEmail,
        )
        result = response.choices[0].message.parsed
        if not result:
            logger.error("No result returned from spam filter LLM.")
            raise ValueError(f"The value of filtered email is {result}")
        logger.info(
            f"Spam filtering completed: is_spam={result.is_spam}, confidence_score={result.confidence_score}"
        )
        return result
    except Exception as e:
        logger.exception(f"Exception occurred during spam filtering: {e}")
        raise


def get_cleaned_mail(mail: EmailInput) -> CleanedEmailOutput:
    """
    Cleans an email by extracting only the meaningful content.

    Args:
        mail (EmailInput): The email to be cleaned.

    Returns:
        CleanedEmailOutput: The cleaned content of the email.

    Raises:
        Exception: If the LLM call fails or returns no result.
    """
    logger.info(f"Starting cleaning process for email with subject: '{mail.subject}'")
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract and clean just the meaningful content from the mail (remove signatures, disclaimers, quoted replies, etc.).",
                },
                {"role": "user", "content": str(mail)},
            ],
            response_format=CleanedEmailOutput,
        )
        result = response.choices[0].message.parsed
        if not result:
            logger.error("No result returned from cleaning LLM.")
            raise ValueError(f"Value of cleaned email output is {result}")
        logger.info("Email cleaning completed successfully.")
        return result
    except Exception as e:
        logger.exception(f"Exception occurred during email cleaning: {e}")
        raise


def summarise_mail(mail: CleanedEmailOutput) -> EmailSummaryOutput:
    """
    Summarizes the cleaned content of an email.

    Args:
        mail (CleanedEmailOutput): The cleaned email content.

    Returns:
        EmailSummaryOutput: The summary of the email.

    Raises:
        Exception: If the LLM call fails or returns no result.
    """
    logger.info("Starting summarization of cleaned email content.")
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "write a short summary of the given content",
                },
                {"role": "user", "content": str(mail)},
            ],
            response_format=EmailSummaryOutput,
        )
        result = response.choices[0].message.parsed
        if not result:
            logger.error("No result returned from summarization LLM.")
            raise ValueError(f"Value of summarised mail is {result}")
        logger.info("Email summarization completed successfully.")
        return result
    except Exception as e:
        logger.exception(f"Exception occurred during email summarization: {e}")
        raise


def get_summary_of_mail(mail: EmailInput):
    """
    Processes an email through spam filtering, cleaning, and summarization.

    Args:
        mail (EmailInput): The email to process.

    Returns:
        EmailSummaryOutput or None: The summary if not spam and processing succeeds, else None.
    """
    logger.info(f"Initiating summary process for email with subject: '{mail.subject}'")
    try:
        mail_filter = filter_mail(mail)
        if mail_filter.is_spam and mail_filter.confidence_score > 0.70:
            logger.warning(
                f"Email identified as spam with confidence {mail_filter.confidence_score:.2f}. Skipping further processing."
            )
            return None

        logger.info("Email passed spam filter. Proceeding to cleaning step.")
        cleaned_mail = get_cleaned_mail(mail)
        if not cleaned_mail or not cleaned_mail.clean_output.strip():
            logger.error("Cleaned email content is empty. Aborting summarization.")
            return None

        logger.info("Email cleaning step completed. Proceeding to summarization.")
        final_output = summarise_mail(cleaned_mail)
        if not final_output or not final_output.summary.strip():
            logger.error("Summary generation failed or returned empty summary.")
            return None

        logger.info("Summary generation completed successfully.")
        return final_output
    except Exception as e:
        logger.error(f"Failed to generate summary for email: {e}", exc_info=True)
        return None


# Test the email processing pipeline with a sample email
mail = {
    "subject": "Project Meeting Rescheduled to Thursday at 2 PM",
    "sender": "jane.doe@company.com",
    "recipient": "you@example.com",
    "body": "Hi,\n\nJust a quick note to let you know that the meeting originally scheduled for Wednesday has been moved to Thursday at 2 PM in Room 304. Let me know if this works for you.\n\nBest,\nJane",
}

parsed_mail = EmailInput(**mail)

result = get_summary_of_mail(parsed_mail)
if result:
    logger.info(f"Final summary: {result.summary}")
    print(result.summary)
else:
    logger.info("No summary generated for the provided email.")


# Test the spam filtering and summarization pipeline with a spam email
spam_mail = {
    "subject": "Congratulations! You’ve won a $1000 gift card 🎉",
    "sender": "rewards@freestuffnow.biz",
    "recipient": "you@example.com",
    "body": "Dear User,\n\nYou have been selected to receive a FREE $1000 Amazon gift card! Just click the link below to claim your reward. This offer expires soon!\n\n👉 http://scam-link.com\n\nAct fast!\n\n- The Free Stuff Team",
}

parsed_spam_mail = EmailInput(**spam_mail)

result = get_summary_of_mail(parsed_spam_mail)
if result:
    logger.info(f"Final summary: {result.summary}")
    print(result.summary)
else:
    logger.info("No summary generated for the provided email.")
