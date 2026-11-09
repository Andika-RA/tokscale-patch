//! Deep AI Code session parser
//!
//! Deep AI Code is an Electron harness that embeds the Pi coding agent and
//! writes the same session JSONL record layout, so parsing delegates to
//! `super::pi::parse_pi_format_file`; only the scan root and client id differ.
//!
//! Sessions live under the app's user-data tree rather than Pi's own
//! `~/.pi/agent/sessions`, so the two are never the same tree:
//! `~/.deepai-code/userdata/deepai/pi/deepai/sessions/<encoded-cwd>/<timestamp>_<uuid>.jsonl`.
//! The root is deliberately home-relative instead of honoring
//! `PI_CODING_AGENT_DIR`: that variable is shared with Pi itself, and pointing
//! this client at it would make one override feed both clients and double-count
//! the same transcripts. Messages keep their recorded provider (e.g. `deepai`)
//! and model, so a Deep AI Code run is attributed separately from a Pi CLI run.

use super::pi::parse_pi_format_file;
use super::UnifiedMessage;
use std::path::Path;

/// Parse a Deep AI Code JSONL session file.
pub fn parse_deepai_file(path: &Path) -> Vec<UnifiedMessage> {
    parse_pi_format_file(path, "deepai", "deepai")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_file(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_parse_deepai_session_stamps_deepai_client() {
        // given: the record layout Deep AI Code writes — a versioned session
        // header followed by assistant messages carrying nested usage.
        let content = r#"{"type":"session","version":3,"id":"01a1be49-44ff-7182-9ba7-4b9e94551dc3","timestamp":"2026-11-09T03:10:35.263Z","cwd":"/tmp/workspace"}
{"type":"message","id":"b2","parentId":null,"timestamp":"2026-11-09T03:10:40.000Z","message":{"role":"assistant","model":"glm-5.3-flash","provider":"deepai","usage":{"input":15180,"output":202,"cacheRead":0,"cacheWrite":0,"totalTokens":15382}}}"#;
        let file = create_test_file(content);

        // when
        let messages = parse_deepai_file(file.path());

        // then
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].client, "deepai");
        assert_eq!(
            messages[0].session_id,
            "01a1be49-44ff-7182-9ba7-4b9e94551dc3"
        );
        assert_eq!(messages[0].model_id, "glm-5.3-flash");
        assert_eq!(messages[0].provider_id, "deepai");
        assert_eq!(messages[0].tokens.input, 15180);
        assert_eq!(messages[0].tokens.output, 202);
    }

    #[test]
    fn test_parse_deepai_missing_provider_falls_back_to_deepai() {
        // given: a transcript whose assistant records carry no provider field.
        let content = r#"{"type":"session","version":3,"id":"01a1c069-98e9-712c-b03e-106af603d3a9","timestamp":"2026-11-09T13:05:08.330Z","cwd":"/tmp/proj"}
{"type":"message","id":"m1","parentId":null,"timestamp":"2026-11-09T13:05:20.000Z","message":{"role":"assistant","model":"acme/custom-chip-model-1","usage":{"input":900,"output":40,"cacheRead":100,"cacheWrite":0,"totalTokens":1040}}}"#;
        let file = create_test_file(content);

        // when
        let messages = parse_deepai_file(file.path());

        // then
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].provider_id, "deepai");
        assert_eq!(messages[0].tokens.cache_read, 100);
    }

    #[test]
    fn test_parse_deepai_skips_non_assistant_records() {
        // given: a user record that carries usage — only assistant calls bill.
        let content = r#"{"type":"session","version":3,"id":"s1","timestamp":"2026-11-09T00:00:00.000Z","cwd":"/tmp"}
{"type":"message","timestamp":"2026-11-09T00:00:01.000Z","message":{"role":"user","model":"glm-5.3-flash","provider":"deepai","usage":{"input":10,"output":5,"cacheRead":0,"cacheWrite":0,"totalTokens":15}}}"#;
        let file = create_test_file(content);

        // when
        let messages = parse_deepai_file(file.path());

        // then
        assert!(messages.is_empty());
    }
}
