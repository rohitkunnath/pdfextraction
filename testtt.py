import pytest
from unittest.mock import MagicMock, patch
import asyncio
from main2 import start_chat, handle_message  # Replace `your_module_name` with the actual module name

@pytest.fixture
def mock_file():
    file_mock = MagicMock()
    file_mock.name = "test.pdf"
    file_mock.path = "path/to/test.pdf"
    return file_mock

@pytest.fixture
def mock_chain():
    chain_mock = MagicMock()
    chain_mock.ainvoke.return_value = {
        "answer": "Test answer",
        "source_documents": [
            MagicMock(page_content="Document content 1"),
            MagicMock(page_content="Document content 2")
        ]
    }
    return chain_mock

@pytest.fixture
def mock_cl():
    with patch('chainlit.AskFileMessage') as ask_file_msg:
        with patch('chainlit.Message') as message_cls:
            yield ask_file_msg, message_cls

@pytest.mark.asyncio
async def test_start_chat(mock_file, mock_cl):
    ask_file_msg, message_cls = mock_cl
    ask_file_msg.return_value.send.return_value = [mock_file]

    with patch('PyPDF2.PdfReader') as pdf_reader:
        pdf_reader.return_value.pages = [MagicMock(extract_text=MagicMock(return_value="Sample text"))]
        pdf_reader.return_value.pages[0].extract_text.return_value = "Sample text"

        with patch('your_module_name.Chroma.from_texts') as from_texts, \
             patch('your_module_name.OllamaEmbeddings') as embeddings, \
             patch('your_module_name.ConversationalRetrievalChain.from_llm') as from_llm:

            from_texts.return_value.as_retriever.return_value = MagicMock()
            embeddings.return_value = MagicMock()
            from_llm.return_value = MagicMock()
            from_llm.return_value.as_retriever.return_value = MagicMock()

            await start_chat()

            # Check that the processing message was sent
            message_cls.assert_called_with(content="Processing `test.pdf` done. You can now ask questions!")

@pytest.mark.asyncio
async def test_handle_message(mock_chain, mock_cl):
    ask_file_msg, message_cls = mock_cl
    chain_mock = mock_chain

    with patch('your_module_name.cl.user_session.get') as get_session:
        get_session.return_value = chain_mock

        test_message = MagicMock()
        test_message.content = "Test query"

        await handle_message(test_message)

        # Check that the message with the answer was sent
        message_cls.assert_called_with(content="Test answer\nSources: source_0, source_1")

if __name__ == "__main__":
    pytest.main()
