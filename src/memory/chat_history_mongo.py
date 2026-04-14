"""
Chat history storage using MongoDB backend.
"""

from datetime import datetime
import logging
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from src.db.mongo_client import db

collection = db["chat_history"]
logger = logging.getLogger(__name__)


class MongoDBChatMessageHistory(BaseChatMessageHistory):
    """Chat history backed by MongoDB with in-memory fallback."""

    _fallback_store: dict[str, ChatMessageHistory] = {}

    def __init__(self, session_id: str):
        """
        Initialize chat history for a session.

        Args:
            session_id: Unique session identifier.
        """
        self.session_id = session_id
        self._mongo_available = True

    def _fallback_history(self) -> ChatMessageHistory:
        """Get or create in-memory chat history for this session."""
        if self.session_id not in self._fallback_store:
            self._fallback_store[self.session_id] = ChatMessageHistory()
        return self._fallback_store[self.session_id]

    def _disable_mongo(self, error: Exception) -> None:
        """Disable Mongo usage for this session after a connection failure."""
        if self._mongo_available:
            logger.warning(
                "MongoDB unavailable for session %s. Falling back to in-memory history. Error: %s",
                self.session_id,
                error,
            )
        self._mongo_available = False

    async def add_message(self, message: BaseMessage) -> None:
        """
        Save a message to MongoDB.

        Args:
            message: The message to save.
        """
        if not self._mongo_available:
            self._fallback_history().add_message(message)
            return

        try:
            await collection.insert_one({
                "session_id": self.session_id,
                "type": message.type,
                "content": message.content,
                "additional_kwargs": message.additional_kwargs,
                "timestamp": datetime.utcnow(),
            })
        except Exception as exc:
            self._disable_mongo(exc)
            self._fallback_history().add_message(message)

    async def get_messages(self) -> List[BaseMessage]:
        """
        Load all messages for a session from MongoDB.

        Returns:
            List of messages in chronological order.
        """
        from langchain_core.messages import messages_from_dict

        if not self._mongo_available:
            return list(self._fallback_history().messages)

        try:
            cursor = collection.find({"session_id": self.session_id}).sort("timestamp", 1)
            docs = await cursor.to_list(length=1000)

            return messages_from_dict([
                {
                    "type": d["type"],
                    "data": {
                        "content": d["content"],
                        "additional_kwargs": d.get("additional_kwargs", {}),
                    }
                }
                for d in docs
            ])
        except Exception as exc:
            self._disable_mongo(exc)
            return list(self._fallback_history().messages)

    async def clear(self) -> None:
        """Delete all messages for a session."""
        if self._mongo_available:
            try:
                await collection.delete_many({"session_id": self.session_id})
            except Exception as exc:
                self._disable_mongo(exc)

        self._fallback_history().clear()


class ChatHistory:
    """Factory for MongoDB-backed chat history."""

    @classmethod
    def get_session_history(
        cls,
        session_id: str,
        config: dict = None
    ) -> BaseChatMessageHistory:
        """
        Get or create chat history for a session.

        Args:
            session_id: Unique session identifier.
            config: Optional configuration dictionary.

        Returns:
            Chat history instance for the session.
        """
        return MongoDBChatMessageHistory(session_id)
