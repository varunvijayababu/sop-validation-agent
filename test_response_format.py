import sys
import os
from unittest.mock import MagicMock, patch

# Add workspace root to sys.path dynamically
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock the inner SDK clients before importing adapters
# Groq client mock
mock_groq_completions = MagicMock()
mock_groq_response = MagicMock()
mock_groq_response.choices = [MagicMock()]
mock_groq_response.choices[0].message.content = "Groq output content"
mock_groq_response.usage = MagicMock()
mock_groq_response.usage.prompt_tokens = 100
mock_groq_response.usage.completion_tokens = 50
mock_groq_response.usage.total_tokens = 150
mock_groq_completions.create = MagicMock(return_value=mock_groq_response)

# OpenAI client mock
mock_openai_completions = MagicMock()
mock_openai_response = MagicMock()
mock_openai_response.choices = [MagicMock()]
mock_openai_response.choices[0].message.content = "OpenAI output content"
mock_openai_response.usage = MagicMock()
mock_openai_response.usage.prompt_tokens = 200
mock_openai_response.usage.completion_tokens = 80
mock_openai_response.usage.total_tokens = 280
mock_openai_completions.create = MagicMock(return_value=mock_openai_response)

# Gemini client mock
mock_gemini_models = MagicMock()
mock_gemini_response = MagicMock()
mock_gemini_response.text = "Gemini output content"
mock_gemini_response.usage_metadata = MagicMock()
mock_gemini_response.usage_metadata.prompt_token_count = 300
mock_gemini_response.usage_metadata.candidates_token_count = 120
mock_gemini_response.usage_metadata.total_token_count = 420
mock_gemini_models.generate_content = MagicMock(return_value=mock_gemini_response)

# Ollama client mock
mock_ollama_response = {
    "message": {
        "role": "assistant",
        "content": "Ollama output content"
    },
    "prompt_eval_count": 400,
    "eval_count": 150
}

# Import LLM components first
from app.llm.groq_adapter import GroqAdapter
from app.llm.openai_adapter import OpenAIAdapter
from app.llm.gemini_adapter import GeminiAdapter
from app.llm.ollama_adapter import OllamaAdapter
import app.llm.config as llm_config

# Patch the specific modules' imported client classes
with patch("app.llm.groq_adapter.Groq") as mock_groq_class, \
     patch("app.llm.openai_adapter.OpenAI") as mock_openai_class, \
     patch("app.llm.gemini_adapter.genai.Client") as mock_gemini_client_class, \
     patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class, \
     patch("app.llm.openai_adapter.OPENAI_API_KEY", "mock-key"), \
     patch("app.llm.gemini_adapter.GEMINI_API_KEY", "mock-key"):
     
    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions = mock_groq_completions
    mock_groq_class.return_value = mock_groq_client

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions = mock_openai_completions
    mock_openai_class.return_value = mock_openai_client

    mock_gemini_client = MagicMock()
    mock_gemini_client.models = mock_gemini_models
    mock_gemini_client_class.return_value = mock_gemini_client

    mock_ollama_client = MagicMock()
    mock_ollama_client.chat = MagicMock(return_value=mock_ollama_response)
    mock_ollama_client_class.return_value = mock_ollama_client

    # Test GroqAdapter directly
    groq_adapter = GroqAdapter()
    groq_res = groq_adapter.generate([{"role": "user", "content": "test"}])
    assert groq_res.content == "Groq output content"
    assert groq_res.token_usage == {"INPUT": 100, "OUTPUT": 50, "TOTAL": 150}
    print("Groq adapter direct test passed!")

    # Test OpenAIAdapter directly
    openai_adapter = OpenAIAdapter()
    openai_res = openai_adapter.generate([{"role": "user", "content": "test"}])
    assert openai_res.content == "OpenAI output content"
    assert openai_res.token_usage == {"INPUT": 200, "OUTPUT": 80, "TOTAL": 280}
    print("OpenAI adapter direct test passed!")

    # Test GeminiAdapter directly
    gemini_adapter = GeminiAdapter()
    gemini_res = gemini_adapter.generate([{"role": "user", "content": "test"}])
    assert gemini_res.content == "Gemini output content"
    assert gemini_res.token_usage == {"INPUT": 300, "OUTPUT": 120, "TOTAL": 420}
    print("Gemini adapter direct test passed!")

    # Test OllamaAdapter directly
    ollama_adapter = OllamaAdapter()
    ollama_res = ollama_adapter.generate([{"role": "user", "content": "test"}])
    assert ollama_res.content == "Ollama output content"
    assert ollama_res.token_usage == {"INPUT": 400, "OUTPUT": 150, "TOTAL": 550}
    print("Ollama adapter direct test passed!")

    # Test Factory selection
    with patch("app.llm.config.LLM_PROVIDER", "groq"), patch("app.llm.factory.LLM_PROVIDER", "groq"):
        from app.llm.factory import get_llm_provider
        resolved_provider = get_llm_provider()
        assert isinstance(resolved_provider, GroqAdapter)
        print("Factory groq selection test passed!")

    with patch("app.llm.config.LLM_PROVIDER", "openai"), patch("app.llm.factory.LLM_PROVIDER", "openai"):
        resolved_provider = get_llm_provider()
        assert isinstance(resolved_provider, OpenAIAdapter)
        print("Factory openai selection test passed!")

    with patch("app.llm.config.LLM_PROVIDER", "gemini"), patch("app.llm.factory.LLM_PROVIDER", "gemini"):
        resolved_provider = get_llm_provider()
        assert isinstance(resolved_provider, GeminiAdapter)
        print("Factory gemini selection test passed!")

    with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
        resolved_provider = get_llm_provider()
        assert isinstance(resolved_provider, OllamaAdapter)
        print("Factory ollama selection test passed!")

# --- Now run general validation flow integration tests with a mock provider ---
import app.agents.llm_validator as validator
from app.llm.base import LLMResponse

mock_response = LLMResponse(
    content="""
[
  {
    "STATUS": "MODIFY",
    "SECTION_RESULTS": [
      {
        "SECTION": "Section A",
        "STATUS": "PARTIAL"
      },
      {
        "SECTION": "Section B",
        "STATUS": "COMPLIANT"
      }
    ],
    "COMMENTS": "Mock comments detailing section status.",
    "REFERENCE": "Section A (Page 2)"
  }
]
""",
    token_usage={
        "INPUT": 1200,
        "OUTPUT": 350,
        "TOTAL": 1550
    }
)

mock_provider = MagicMock()
mock_provider.generate = MagicMock(return_value=mock_response)
validator._cached_validate_sop_internal.cache_clear()

reference_context = [
    {"section": "Section A", "page": 2, "weight": 40.0, "text": "Content A"},
    {"section": "Section B", "page": 3, "weight": 60.0, "text": "Content B"}
]

# Temporarily patch validator.llm / get_llm_provider for basic success tests
with patch("app.agents.llm_validator.get_llm_provider", return_value=mock_provider):
    res_simple = validator.validate_sop("Dummy SOP text", reference_context, detailed=False)
    res_detailed = validator.validate_sop("Dummy SOP text", reference_context, detailed=True)

# Assertions for response shapes & keys
item_simple = res_simple[0]
keys_simple = list(item_simple.keys())
assert keys_simple == ["STATUS", "SCORE", "COMMENTS", "REFERENCE"]
assert "TOKEN_COUNT" not in item_simple

item_detailed = res_detailed[0]
keys_detailed = list(item_detailed.keys())
assert keys_detailed == ["STATUS", "SCORE", "SCORE_BREAKDOWN", "COMMENTS", "REFERENCE", "TOKEN_COUNT"]
assert item_detailed["TOKEN_COUNT"] == {
    "INPUT": 1200,
    "OUTPUT": 350,
    "TOTAL": 1550
}

# Verify cache hit and overall score matching
assert mock_provider.generate.call_count == 1
assert item_simple["STATUS"] == item_detailed["STATUS"]
assert item_simple["SCORE"] == item_detailed["SCORE"]

breakdown_sum = sum(sec["SCORE"] for sec in item_detailed["SCORE_BREAKDOWN"].values())
assert abs(breakdown_sum - item_detailed["SCORE"]) < 1e-9

# --- Provider-Aware Error Mapping Tests ---
print("Running provider-aware error mapping tests...")

# 1. Groq rate limit mapping
with patch("app.llm.config.LLM_PROVIDER", "groq"), patch("app.llm.factory.LLM_PROVIDER", "groq"):
    validator._cached_validate_sop_internal.cache_clear()
    import groq
    with patch("app.llm.groq_adapter.Groq") as mock_groq_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = groq.RateLimitError(
            message="Rate limit reached",
            response=MagicMock(),
            body=None
        )
        mock_groq_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=False)
        assert res[0]["STATUS"] == "SYSTEM_ERROR"
        assert "Groq request failed: rate limit exceeded." in res[0]["COMMENTS"], f"Expected Groq rate limit message, got: {res[0]['COMMENTS']}"
        print("Groq rate limit exception mapping test passed!")

# 2. OpenAI insufficient quota mapping
with patch("app.llm.config.LLM_PROVIDER", "openai"), patch("app.llm.factory.LLM_PROVIDER", "openai"), patch("app.llm.openai_adapter.OPENAI_API_KEY", "mock-key"):
    validator._cached_validate_sop_internal.cache_clear()
    import openai
    with patch("app.llm.openai_adapter.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="insufficient_quota: Please check your API billing",
            response=MagicMock(),
            body=None
        )
        mock_openai_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=False)
        assert res[0]["STATUS"] == "SYSTEM_ERROR"
        assert "OpenAI request failed: insufficient quota. Please check your API billing or usage limits." in res[0]["COMMENTS"], f"Expected OpenAI quota message, got: {res[0]['COMMENTS']}"
        print("OpenAI insufficient quota exception mapping test passed!")

# 3. Gemini APIError mapping
with patch("app.llm.config.LLM_PROVIDER", "gemini"), patch("app.llm.factory.LLM_PROVIDER", "gemini"), patch("app.llm.gemini_adapter.GEMINI_API_KEY", "mock-key"):
    validator._cached_validate_sop_internal.cache_clear()
    from google.genai import errors
    with patch("app.llm.gemini_adapter.genai.Client") as mock_gemini_client_class:
        mock_client = MagicMock()
        api_error = errors.APIError(
            code=429,
            response_json={'error': {'message': 'Resource has been exhausted'}}
        )
        mock_client.models.generate_content.side_effect = api_error
        mock_gemini_client_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=False)
        assert res[0]["STATUS"] == "SYSTEM_ERROR"
        assert "Gemini request failed: rate limit or quota exceeded. Please check your billing or retry later." in res[0]["COMMENTS"], f"Expected Gemini rate limit message, got: {res[0]['COMMENTS']}"
        print("Gemini APIError rate limit exception mapping test passed!")

# 4. Ollama connection mapping
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    validator._cached_validate_sop_internal.cache_clear()
    with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
        mock_client = MagicMock()
        mock_client.chat.side_effect = Exception("Connection refused by localhost:11434")
        mock_ollama_client_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=False)
        assert res[0]["STATUS"] == "SYSTEM_ERROR"
        assert "Ollama request failed: server unavailable or connection refused" in res[0]["COMMENTS"], f"Expected Ollama connection error, got: {res[0]['COMMENTS']}"
        print("Ollama connection exception mapping test passed!")

# 5. Ollama missing model mapping
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    validator._cached_validate_sop_internal.cache_clear()
    with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
        mock_client = MagicMock()
        mock_client.chat.side_effect = Exception("model 'llama3' not found, try pulling it")
        mock_ollama_client_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=False)
        assert res[0]["STATUS"] == "SYSTEM_ERROR"
        assert "Ollama request failed: model 'llama3' not found on server." in res[0]["COMMENTS"], f"Expected Ollama model 404 error, got: {res[0]['COMMENTS']}"
        print("Ollama model not found exception mapping test passed!")

# 6. Fallback default factory failure mapping
with patch("app.llm.config.LLM_PROVIDER", "unsupported-provider"), patch("app.llm.factory.LLM_PROVIDER", "unsupported-provider"):
    validator._cached_validate_sop_internal.cache_clear()
    res = validator.validate_sop("Dummy SOP text", reference_context, detailed=False)
    assert res[0]["STATUS"] == "SYSTEM_ERROR"
    assert "Unsupported-provider request failed: Unsupported LLM provider: 'unsupported-provider'" in res[0]["COMMENTS"], f"Expected fallback provider factory error, got: {res[0]['COMMENTS']}"
    print("Fallback provider factory failure mapping test passed!")

# 7. Empty response JSON parsing error mapping
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    validator._cached_validate_sop_internal.cache_clear()
    with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "   " # Whitespace only
            }
        }
        mock_ollama_client_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=False)
        assert res[0]["STATUS"] == "SYSTEM_ERROR"
        assert "JSON Parsing Error: Model returned an empty or whitespace-only response" in res[0]["COMMENTS"], f"Expected empty response message, got: {res[0]['COMMENTS']}"
        print("Empty response error mapping test passed!")

# 8. Detailed error response mapping (verifying zeroed TOKEN_COUNT)
with patch("app.llm.config.LLM_PROVIDER", "unsupported-provider"), patch("app.llm.factory.LLM_PROVIDER", "unsupported-provider"):
    validator._cached_validate_sop_internal.cache_clear()
    res = validator.validate_sop("Dummy SOP text", reference_context, detailed=True)
    assert res[0]["STATUS"] == "SYSTEM_ERROR"
    assert res[0]["TOKEN_COUNT"] == {"INPUT": 0, "OUTPUT": 0, "TOTAL": 0}
    assert list(res[0].keys()) == ["STATUS", "SCORE", "SCORE_BREAKDOWN", "COMMENTS", "REFERENCE", "TOKEN_COUNT"]
    print("Detailed fallback provider error mapping test passed!")

# 9. JSON object wrapped into single-item list format & Prose-wrapped JSON extraction
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    validator._cached_validate_sop_internal.cache_clear()
    with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": """
Here is the raw result:
{
  "STATUS": "ACCEPT",
  "SECTION_RESULTS": [
    {
      "SECTION": "Section A",
      "STATUS": "COMPLIANT"
    },
    {
      "SECTION": "Section B",
      "STATUS": "COMPLIANT"
    }
  ],
  "COMMENTS": "The document is fully compliant with all guidelines.",
  "REFERENCE": "Section A (Page 2)"
}
Enjoy!
"""
            }
        }
        mock_ollama_client_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=False)
        assert res[0]["STATUS"] == "ACCEPT"
        assert res[0]["COMMENTS"] == "The document is fully compliant with all guidelines."
        print("Prose-wrapped JSON object extraction and normalization test passed!")

# 10. Bad top-level STATUS rejection
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    validator._cached_validate_sop_internal.cache_clear()
    with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": """
[
  {
    "STATUS": "UNKNOWN_STATUS",
    "SECTION_RESULTS": [],
    "COMMENTS": "Fully compliant.",
    "REFERENCE": "Section A (Page 2)"
  }
]
"""
            }
        }
        mock_ollama_client_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=False)
        assert res[0]["STATUS"] == "SYSTEM_ERROR"
        assert "Invalid top-level STATUS" in res[0]["COMMENTS"]
        print("Bad top-level STATUS rejection test passed!")

# 11. Placeholder value rejection
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    validator._cached_validate_sop_internal.cache_clear()
    with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": """
[
  {
    "STATUS": "ACCEPT",
    "SECTION_RESULTS": [
      {
        "SECTION": "Section Name",
        "STATUS": "COMPLIANT"
      }
    ],
    "COMMENTS": "The document is fully compliant.",
    "REFERENCE": "Section A (Page 2)"
  }
]
"""
            }
        }
        mock_ollama_client_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=False)
        assert res[0]["STATUS"] == "SYSTEM_ERROR"
        assert "Placeholder SECTION name detected" in res[0]["COMMENTS"]
        print("Placeholder value rejection test passed!")

# 12. Strict case, punctuation, whitespace differences & title-only unique mapping (should pass and normalize)
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    validator._cached_validate_sop_internal.cache_clear()
    with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": """
[
  {
    "STATUS": "accept",
    "SECTION_RESULTS": [
      {
        "SECTION": "section   a:",
        "STATUS": "compliant"
      },
      {
        "SECTION": "SECTION B",
        "STATUS": "compliant"
      }
    ],
    "COMMENTS": "Looks good.",
    "REFERENCE": "section a"
  }
]
"""
            }
        }
        mock_ollama_client_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=False)
        assert res[0]["STATUS"] == "ACCEPT"
        assert res[0]["REFERENCE"] == "Section A (Page 2)"
        assert res[0]["SCORE"] == 100.0
        print("Conservative case/punctuation/whitespace/title-only resolving test passed!")

# 13. Synonym/paraphrase rejection (should fail with SYSTEM_ERROR)
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    validator._cached_validate_sop_internal.cache_clear()
    with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": """
[
  {
    "STATUS": "ACCEPT",
    "SECTION_RESULTS": [
      {
        "SECTION": "Section A",
        "STATUS": "COMPLIANT"
      },
      {
        "SECTION": "Section B",
        "STATUS": "COMPLIANT"
      }
    ],
    "COMMENTS": "Looks good.",
    "REFERENCE": "Infection Prevention Policy (Page 2)"
  }
]
"""
            }
        }
        mock_ollama_client_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=False)
        assert res[0]["STATUS"] == "SYSTEM_ERROR"
        assert "Invalid REFERENCE" in res[0]["COMMENTS"]
        print("Synonym/paraphrase reference rejection test passed!")

# 14. Ambiguous normalized match rejection (should fail with SYSTEM_ERROR)
ambiguous_reference_context = [
    {"section": "Section Alpha", "page": 2, "weight": 40.0, "text": "Content A"},
    {"section": "Section alpha", "page": 3, "weight": 60.0, "text": "Content B"}
]
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    validator._cached_validate_sop_internal.cache_clear()
    with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": """
[
  {
    "STATUS": "ACCEPT",
    "SECTION_RESULTS": [
      {
        "SECTION": "Section Alpha",
        "STATUS": "COMPLIANT"
      },
      {
        "SECTION": "Section alpha",
        "STATUS": "COMPLIANT"
      }
    ],
    "COMMENTS": "Looks good.",
    "REFERENCE": "section alpha"
  }
]
"""
            }
        }
        mock_ollama_client_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", ambiguous_reference_context, detailed=False)
        assert res[0]["STATUS"] == "SYSTEM_ERROR"
        assert "Ambiguous reference" in res[0]["COMMENTS"] or "Ambiguous section" in res[0]["COMMENTS"]
        print("Ambiguous normalized match rejection test passed!")

# 15. Unrelated invented section rejection (should fail with SYSTEM_ERROR)
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    validator._cached_validate_sop_internal.cache_clear()
    with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": """
[
  {
    "STATUS": "ACCEPT",
    "SECTION_RESULTS": [
      {
        "SECTION": "Section A",
        "STATUS": "COMPLIANT"
      },
      {
        "SECTION": "Invented Section Name",
        "STATUS": "COMPLIANT"
      }
    ],
    "COMMENTS": "Looks good.",
    "REFERENCE": "Section A (Page 2)"
  }
]
"""
            }
        }
        mock_ollama_client_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=False)
        assert res[0]["STATUS"] == "SYSTEM_ERROR"
        assert "Section name 'Invented Section Name' could not be resolved" in res[0]["COMMENTS"]
        print("Unrelated invented section rejection test passed!")

# 16. Page-suffixed section exact match & case/whitespace/punctuation drift
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    validator._cached_validate_sop_internal.cache_clear()
    with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": """
[
  {
    "STATUS": "accept",
    "SECTION_RESULTS": [
      {
        "SECTION": "Section A (Page 2)",
        "STATUS": "compliant"
      },
      {
        "SECTION": "section   b: (page 3)",
        "STATUS": "compliant"
      }
    ],
    "COMMENTS": "Looks good.",
    "REFERENCE": "Section A (Page 2)"
  }
]
"""
            }
        }
        mock_ollama_client_class.return_value = mock_client
        
        # Test both detailed=True and detailed=False
        res_simple = validator.validate_sop("Dummy SOP text", reference_context, detailed=False)
        assert res_simple[0]["STATUS"] == "ACCEPT"
        assert res_simple[0]["SCORE"] == 100.0
        
        res_detailed = validator.validate_sop("Dummy SOP text", reference_context, detailed=True)
        assert res_detailed[0]["STATUS"] == "ACCEPT"
        assert res_detailed[0]["TOKEN_COUNT"] is not None
        assert "Section A" in res_detailed[0]["SCORE_BREAKDOWN"]
        assert "Section B" in res_detailed[0]["SCORE_BREAKDOWN"]
        print("Page-suffixed section name with casing/whitespace/punctuation drift normalized test passed!")

# 17. Invalid paraphrased page-suffixed section must fail
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    validator._cached_validate_sop_internal.cache_clear()
    with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": """
[
  {
    "STATUS": "ACCEPT",
    "SECTION_RESULTS": [
      {
        "SECTION": "Section A (Page 2)",
        "STATUS": "COMPLIANT"
      },
      {
        "SECTION": "Paraphrased Section Name (Page 3)",
        "STATUS": "COMPLIANT"
      }
    ],
    "COMMENTS": "Looks good.",
    "REFERENCE": "Section A (Page 2)"
  }
]
"""
            }
        }
        mock_ollama_client_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=False)
        assert res[0]["STATUS"] == "SYSTEM_ERROR"
        assert "Section name 'Paraphrased Section Name (Page 3)' could not be resolved" in res[0]["COMMENTS"]
        print("Invalid paraphrased page-suffixed section name rejection test passed!")

# 18. Successful Retry validation (Ollama)
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    validator._cached_validate_sop_internal.cache_clear()
    with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
        mock_client = MagicMock()
        
        # Define response 1 (partial evaluation: only 1 section result out of 2)
        response_1 = {
            "message": {
                "role": "assistant",
                "content": """
[
  {
    "STATUS": "ACCEPT",
    "SECTION_RESULTS": [
      {
        "SECTION": "Section A",
        "STATUS": "COMPLIANT"
      }
    ],
    "COMMENTS": "Looks good.",
    "REFERENCE": "Section A (Page 2)"
  }
]
"""
            },
            "prompt_eval_count": 100,
            "eval_count": 50
        }
        
        # Define response 2 (complete evaluation: both Section A and Section B)
        response_2 = {
            "message": {
                "role": "assistant",
                "content": """
[
  {
    "STATUS": "ACCEPT",
    "SECTION_RESULTS": [
      {
        "SECTION": "Section A",
        "STATUS": "COMPLIANT"
      },
      {
        "SECTION": "Section B",
        "STATUS": "COMPLIANT"
      }
    ],
    "COMMENTS": "Looks good.",
    "REFERENCE": "Section A (Page 2)"
  }
]
"""
            },
            "prompt_eval_count": 200,
            "eval_count": 80
        }
        
        # Set side effect
        mock_client.chat.side_effect = [response_1, response_2]
        mock_ollama_client_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=True)
        assert res[0]["STATUS"] == "ACCEPT"
        assert res[0]["SCORE"] == 100.0
        # Verification of retry logic having run twice
        assert mock_client.chat.call_count == 2
        
        # Verification of token counts across retry attempts
        assert res[0]["TOKEN_COUNT"]["INPUT"] == 300  # 100 + 200
        assert res[0]["TOKEN_COUNT"]["OUTPUT"] == 130 # 50 + 80
        assert res[0]["TOKEN_COUNT"]["TOTAL"] == 430  # 300 + 130
        print("Ollama successful retry validation & token count aggregation test passed!")

# 19. Unsuccessful Retry validation (Ollama)
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    validator._cached_validate_sop_internal.cache_clear()
    with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
        mock_client = MagicMock()
        
        # Both attempts return incomplete section evaluation
        bad_response = {
            "message": {
                "role": "assistant",
                "content": """
[
  {
    "STATUS": "ACCEPT",
    "SECTION_RESULTS": [
      {
        "SECTION": "Section A",
        "STATUS": "COMPLIANT"
      }
    ],
    "COMMENTS": "Looks good.",
    "REFERENCE": "Section A (Page 2)"
  }
]
"""
            },
            "prompt_eval_count": 120,
            "eval_count": 40
        }
        
        mock_client.chat.side_effect = [bad_response, bad_response]
        mock_ollama_client_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=True)
        assert res[0]["STATUS"] == "SYSTEM_ERROR"
        assert "Ollama returned 1 section results, but expected 2." in res[0]["COMMENTS"]
        assert mock_client.chat.call_count == 2
        # Verification of token counts across retry attempts (detailed=True)
        assert res[0]["TOKEN_COUNT"]["INPUT"] == 240
        assert res[0]["TOKEN_COUNT"]["OUTPUT"] == 80
        assert res[0]["TOKEN_COUNT"]["TOTAL"] == 320
        print("Ollama unsuccessful retry validation & token count aggregation test passed!")

# 20. No Retry validation (Ollama)
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    validator._cached_validate_sop_internal.cache_clear()
    with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
        mock_client = MagicMock()
        
        good_response = {
            "message": {
                "role": "assistant",
                "content": """
[
  {
    "STATUS": "ACCEPT",
    "SECTION_RESULTS": [
      {
        "SECTION": "Section A",
        "STATUS": "COMPLIANT"
      },
      {
        "SECTION": "Section B",
        "STATUS": "COMPLIANT"
      }
    ],
    "COMMENTS": "Looks good.",
    "REFERENCE": "Section A (Page 2)"
  }
]
"""
            },
            "prompt_eval_count": 150,
            "eval_count": 70
        }
        
        mock_client.chat.return_value = good_response
        mock_ollama_client_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=True)
        assert res[0]["STATUS"] == "ACCEPT"
        assert mock_client.chat.call_count == 1
        assert res[0]["TOKEN_COUNT"]["INPUT"] == 150
        assert res[0]["TOKEN_COUNT"]["OUTPUT"] == 70
        assert res[0]["TOKEN_COUNT"]["TOTAL"] == 220
        print("Ollama no retry validation test passed!")

# 21. Groq regression check (No retry, fallback missing-section behavior)
with patch("app.llm.config.LLM_PROVIDER", "groq"), patch("app.llm.factory.LLM_PROVIDER", "groq"):
    validator._cached_validate_sop_internal.cache_clear()
    # Mock a response that returns only 1 section for a 2-section reference context.
    # For Groq, it should NOT retry, but instead autofill the missing section as MISSING.
    mock_groq_response = MagicMock()
    mock_groq_response.choices = [MagicMock()]
    mock_groq_response.choices[0].message.content = """
[
  {
    "STATUS": "ACCEPT",
    "SECTION_RESULTS": [
      {
        "SECTION": "Section A",
        "STATUS": "COMPLIANT"
      }
    ],
    "COMMENTS": "Looks good.",
    "REFERENCE": "Section A (Page 2)"
  }
]
"""
    mock_groq_response.usage = MagicMock()
    mock_groq_response.usage.prompt_tokens = 100
    mock_groq_response.usage.completion_tokens = 50
    mock_groq_response.usage.total_tokens = 150
    
    with patch("app.llm.groq_adapter.Groq") as mock_groq_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_groq_response
        mock_groq_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=True)
        assert res[0]["STATUS"] == "ACCEPT"
        # Section count should be autofilled to 2
        assert len(res[0]["SCORE_BREAKDOWN"]) == 2
        assert res[0]["SCORE_BREAKDOWN"]["Section B"]["STATUS"] == "MISSING"
        # Call count must be exactly 1
        assert mock_client.chat.completions.create.call_count == 1
        print("Groq regression check (no retry & autofill behavior) test passed!")

# 22. Ollama Status Normalization: COMPLETE -> COMPLIANT, INCOMPLETE -> PARTIAL, mixed list
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    validator._cached_validate_sop_internal.cache_clear()
    with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
        mock_client = MagicMock()
        good_response = {
            "message": {
                "role": "assistant",
                "content": """
[
  {
    "STATUS": "ACCEPT",
    "SECTION_RESULTS": [
      {
        "SECTION": "Section A",
        "STATUS": "COMPLETE"
      },
      {
        "SECTION": "Section B",
        "STATUS": "INCOMPLETE"
      }
    ],
    "COMMENTS": "Looks good.",
    "REFERENCE": "Section A (Page 2)"
  }
]
"""
            },
            "prompt_eval_count": 150,
            "eval_count": 70
        }
        mock_client.chat.return_value = good_response
        mock_ollama_client_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=True)
        assert res[0]["STATUS"] == "ACCEPT"
        # Check normalization outcomes
        assert res[0]["SCORE_BREAKDOWN"]["Section A"]["STATUS"] == "COMPLIANT"
        assert res[0]["SCORE_BREAKDOWN"]["Section B"]["STATUS"] == "PARTIAL"
        # Verify scores: Section A (weight 40, COMPLIANT -> score 40), Section B (weight 60, PARTIAL -> score 30). Total score = 70.
        assert res[0]["SCORE"] == 70.0
        print("Ollama status normalization of COMPLETE -> COMPLIANT and INCOMPLETE -> PARTIAL test passed!")

# 23. Ollama Status Normalization: Reject Unknown Statuses
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    for bad_status in ["PASS", "FAIL", "ADEQUATE", "NOT_APPLICABLE", "SOME_INVENTED_STATUS"]:
        validator._cached_validate_sop_internal.cache_clear()
        with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
            mock_client = MagicMock()
            bad_response = {
                "message": {
                    "role": "assistant",
                    "content": f"""
[
  {{
    "STATUS": "ACCEPT",
    "SECTION_RESULTS": [
      {{
        "SECTION": "Section A",
        "STATUS": "{bad_status}"
      }},
      {{
        "SECTION": "Section B",
        "STATUS": "COMPLIANT"
      }}
    ],
    "COMMENTS": "Looks good.",
    "REFERENCE": "Section A (Page 2)"
  }}
]
"""
                },
                "prompt_eval_count": 150,
                "eval_count": 70
            }
            mock_client.chat.side_effect = [bad_response, bad_response]
            mock_ollama_client_class.return_value = mock_client
            
            res = validator.validate_sop("Dummy SOP text", reference_context, detailed=True)
            assert res[0]["STATUS"] == "SYSTEM_ERROR"
            assert f"Invalid section STATUS: {bad_status}" in res[0]["COMMENTS"]
            assert mock_client.chat.call_count == 2
    print("Ollama status normalization rejection of unknown statuses test passed!")

# 24. Ollama Retry success path with normalized statuses
with patch("app.llm.config.LLM_PROVIDER", "ollama"), patch("app.llm.factory.LLM_PROVIDER", "ollama"):
    validator._cached_validate_sop_internal.cache_clear()
    with patch("app.llm.ollama_adapter.Client") as mock_ollama_client_class:
        mock_client = MagicMock()
        response_1 = {
            "message": {
                "role": "assistant",
                "content": """
[
  {
    "STATUS": "ACCEPT",
    "SECTION_RESULTS": [
      {
        "SECTION": "Section A",
        "STATUS": "COMPLETE"
      }
    ],
    "COMMENTS": "Partial count results.",
    "REFERENCE": "Section A (Page 2)"
  }
]
"""
            },
            "prompt_eval_count": 100,
            "eval_count": 50
        }
        response_2 = {
            "message": {
                "role": "assistant",
                "content": """
[
  {
    "STATUS": "ACCEPT",
    "SECTION_RESULTS": [
      {
        "SECTION": "Section A",
        "STATUS": "COMPLETE"
      },
      {
        "SECTION": "Section B",
        "STATUS": "INCOMPLETE"
      }
    ],
    "COMMENTS": "Complete count results with normalized statuses.",
    "REFERENCE": "Section A (Page 2)"
  }
]
"""
            },
            "prompt_eval_count": 200,
            "eval_count": 80
        }
        mock_client.chat.side_effect = [response_1, response_2]
        mock_ollama_client_class.return_value = mock_client
        
        res = validator.validate_sop("Dummy SOP text", reference_context, detailed=True)
        assert res[0]["STATUS"] == "ACCEPT"
        assert res[0]["SCORE"] == 70.0
        assert mock_client.chat.call_count == 2
        print("Ollama retry-success path with normalized statuses test passed!")

# 25. Registry lookup tests for each default provider
from app.llm.base import LLMProvider, LLMResponse
from app.llm.factory import _PROVIDER_REGISTRY, get_llm_provider, register_provider
from app.llm.groq_adapter import GroqAdapter
from app.llm.openai_adapter import OpenAIAdapter
from app.llm.gemini_adapter import GeminiAdapter
from app.llm.ollama_adapter import OllamaAdapter

assert "groq" in _PROVIDER_REGISTRY
assert "openai" in _PROVIDER_REGISTRY
assert "gemini" in _PROVIDER_REGISTRY
assert "ollama" in _PROVIDER_REGISTRY

# Verify case-insensitive, whitespace-trimmed registration
class MockDummyAdapter(LLMProvider):
    @property
    def name(self) -> str: return "MockDummy"
    def map_exception(self, e: Exception) -> str: return "error"
    def generate(self, messages, temperature=0.0) -> LLMResponse:
        return LLMResponse(content="mock", provider_name="MockDummy", model_name="dummy")

register_provider("  Mock-Dummy  ", MockDummyAdapter)
assert "mock-dummy" in _PROVIDER_REGISTRY
assert _PROVIDER_REGISTRY["mock-dummy"] == MockDummyAdapter
print("Registry provider registration test passed!")

# 26. Unsupported provider failure path
with patch("app.llm.config.LLM_PROVIDER", "unsupported-provider"), patch("app.llm.factory.LLM_PROVIDER", "unsupported-provider"):
    try:
        get_llm_provider()
        assert False, "Expected ValueError for unsupported provider"
    except ValueError as e:
        assert "Unsupported LLM provider: 'unsupported-provider'" in str(e)
        print("Registry lookup failure path test passed!")

# 27. Uniform LLMResponse shape across all adapters
with patch("app.llm.groq_adapter.Groq") as mock_groq_class:
    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.choices = [MagicMock()]
    mock_res.choices[0].message.content = "content"
    mock_res.usage = MagicMock()
    mock_client.chat.completions.create.return_value = mock_res
    mock_groq_class.return_value = mock_client
    
    groq_ad = GroqAdapter()
    resp = groq_ad.generate([{"role": "user", "content": "hello"}])
    assert isinstance(resp, LLMResponse)
    assert resp.content == "content"
    assert resp.provider_name == "Groq"
    assert resp.model_name == groq_ad.model
    assert list(resp.token_usage.keys()) == ["INPUT", "OUTPUT", "TOTAL"]
    print("Uniform LLMResponse shape and keys test passed!")

# 28. Environment-variable-only provider switching behavior
with patch("app.llm.config.LLM_PROVIDER", "openai"), patch("app.llm.factory.LLM_PROVIDER", "openai"):
    with patch("app.llm.openai_adapter.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = MagicMock()
        resolved = get_llm_provider()
        assert isinstance(resolved, OpenAIAdapter)
        print("Registry provider switching by environment variable test passed!")

print("ALL SCORING PIPELINE, CACHING, PROVIDER ADAPTERS, AND PROVIDER-AWARE ERROR TESTS PASSED SUCCESSFULLY!")
