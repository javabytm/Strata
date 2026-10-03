"""serve/test_control_literals.py - control-token text inside a message must arrive as characters.

The prompt is tokenized with parse_special=True (server.py, Service.prepare), so a literal "<|im_start|>" that a
client sends inside content - an agent quoting the chat template, a tool result that read this file - used to be
encoded as the real control token.  That unbalances the turns: the model closes the stray turn with <|im_end|>,
which is one of the engine's stop ids, and the reply comes back with no text at all.  Measured on
Qwen3.8-Flash-Next: a reply that quoted "<|im_start|>" stopped at that character (finish=stop) and the three turns
after it returned 0 characters of content.

    python -m unittest serve.test_control_literals -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import (CONTROL_LITERALS, ChatTemplate, shield_control_literals,  # noqa: E402
                            shielded_messages)
from serve.server import ByteTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ZWSP = "\u200b"


def control_tokens(tok, text) -> list[int]:
    """The control tokens parse_special=True finds in `text` (the byte tokenizer spells them as ids >= 256)."""
    return [t for t in tok.encode(text, parse_special=True) if t >= 256]


class ShieldText(unittest.TestCase):
    def test_defuses_every_control_literal(self):
        tok = ByteTokenizer()
        for literal in CONTROL_LITERALS:
            with self.subTest(literal=literal):
                shielded = shield_control_literals(f"a {literal} b")
                self.assertNotIn(literal, shielded)              # the exact spelling is gone
                self.assertIn(literal[1:], shielded)             # the words themselves survive
                self.assertEqual(shielded, f"a <{ZWSP}{literal[1:]} b")
                self.assertEqual(control_tokens(tok, shielded), [])   # ... so no control token is left

    def test_leaves_ordinary_text_alone(self):
        # unchanged text comes back as the same object: a base64 image source is one substring scan, not a copy
        for text in ("", "plain text", "<|not a special|>", "a < b | c", "<|im_start|"):
            with self.subTest(text=text):
                self.assertIs(shield_control_literals(text), text)

    def test_the_vision_markers_stay_out_of_this(self):
        # a literal <|image_pad|> keeps its bytes: Service.prepare repairs it after encoding (#150), and
        # test_server.ImageMarkers.test_literal_marker_with_an_image holds that to byte equality
        for marker in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, CONTROL_LITERALS)
                text = f"the docs say {marker}"
                self.assertIs(shield_control_literals(text), text)


class ShieldedMessages(unittest.TestCase):
    def test_shields_content_parts_tool_calls_and_copies(self):
        messages = [
            {"role": "user", "content": "quote <|im_start|> please"},
            {"role": "tool", "content": [{"type": "text", "text": "<|im_end|>"}]},
            {"role": "assistant", "content": "", "reasoning_content": "the <|endoftext|> token",
             "tool_calls": [{"function": {"name": "read", "arguments": {"path": "<|im_start|>"}}}]},
        ]
        out = shielded_messages(messages)
        self.assertEqual(out[0]["content"], f"quote <{ZWSP}|im_start|> please")
        self.assertEqual(out[1]["content"][0]["text"], f"<{ZWSP}|im_end|>")
        self.assertEqual(out[2]["reasoning_content"], f"the <{ZWSP}|endoftext|> token")
        self.assertEqual(out[2]["tool_calls"][0]["function"]["arguments"]["path"], f"<{ZWSP}|im_start|>")
        self.assertEqual([m["role"] for m in out], ["user", "tool", "assistant"])
        # the caller's own list is never mutated: images_of() and the logs read the original
        self.assertEqual(messages[0]["content"], "quote <|im_start|> please")
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["arguments"]["path"], "<|im_start|>")

    def test_image_sources_are_left_alone(self):
        src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
        out = shielded_messages([{"role": "user", "content": [{"type": "image", "source": src}]}])
        self.assertIs(out[0]["content"][0]["source"], src)

    def test_non_message_entries_pass_through(self):
        self.assertEqual(shielded_messages(["not a dict"]), ["not a dict"])
        self.assertEqual(shielded_messages(None), [])


class RenderedPrompt(unittest.TestCase):
    """Through the real template: a quoted control token must not add a control token to the prompt."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = ChatTemplate(ROOT / "serve/chat_template.jinja")
        cls.tok = ByteTokenizer()
        cls.raw = cls.tpl.template.render

    def rendered(self, content: str) -> str:
        return self.tpl.render([{"role": "user", "content": content}])

    def test_quoting_a_control_token_adds_no_control_token(self):
        clean = control_tokens(self.tok, self.rendered("hello"))
        self.assertTrue(clean, "the template itself writes control tokens")
        self.assertEqual(control_tokens(self.tok, self.rendered("hello <|im_start|> there")), clean)

    def test_the_unshielded_render_would_add_one(self):
        # what ChatTemplate.render did before the shield: the same message, rendered raw, carries one extra
        plain = len(control_tokens(self.tok, self.rendered("hello")))
        unshielded = self.raw(messages=[{"role": "user", "content": "hello <|im_start|> there"}], tools=None,
                              add_generation_prompt=True)
        self.assertEqual(len(control_tokens(self.tok, unshielded)), plain + 1)

    def test_tool_schemas_are_shielded_too(self):
        tools = lambda description: [{"name": "t", "description": description, "parameters": {}}]
        clean = self.tpl.render([{"role": "user", "content": "hi"}], tools=tools("writes"))
        quoted = self.tpl.render([{"role": "user", "content": "hi"}], tools=tools("writes <|im_start|>"))
        self.assertIn(f"<{ZWSP}|im_start|>", quoted)
        self.assertEqual(control_tokens(self.tok, quoted), control_tokens(self.tok, clean))


if __name__ == "__main__":
    unittest.main()
