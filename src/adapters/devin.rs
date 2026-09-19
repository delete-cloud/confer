use std::path::Path;

use anyhow::{Result, bail};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use super::{AdapterOutput, Invocation, cli, error_text, validate_invocation};

pub(super) async fn run(invocation: Invocation, prompt: &str) -> AdapterOutput {
    let export = std::env::temp_dir().join(format!("confer-devin-{}.json", uuid::Uuid::new_v4()));
    let mut command = match build_command(&invocation, prompt, &export) {
        Ok(command) => command,
        Err(error) => return AdapterOutput::failed(error.to_string()),
    };
    let (mut child, stderr) = match super::process::spawn(&mut command, &invocation, false) {
        Ok(process) => process,
        Err(error) => return AdapterOutput::failed(error.to_string()),
    };
    let mut lines = BufReader::new(child.stdout.take().expect("piped stdout")).lines();
    let mut raw_lines = Vec::new();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => raw_lines.push(line),
            Ok(None) => break,
            Err(error) => {
                raw_lines.push(format!("stdout read failed: {error}"));
                break;
            }
        }
    }
    let status = child.wait().await;
    let stderr = String::from_utf8_lossy(&stderr.finish().await).into_owned();
    let exported = read_export(&export);
    let _ = std::fs::remove_file(&export);
    let stdout = raw_lines.join("\n");
    let stdout = stdout.trim();
    let (session, exported_answer) = exported.unwrap_or_default();
    let result = match status {
        Err(error) => Err(format!(
            "failed to wait for {}: {error}",
            invocation.agent.id()
        )),
        Ok(status) if !status.success() => Err(error_text(
            &format!("{} exited with {status}", invocation.agent.id()),
            &stderr,
            stdout,
        )),
        _ => exported_answer
            .or_else(|| (!stdout.is_empty()).then(|| stdout.to_owned()))
            .ok_or_else(|| error_text("agent returned no final answer", &stderr, "")),
    };
    AdapterOutput::from_result(session, result)
}

pub(super) fn build_command(
    invocation: &Invocation,
    prompt: &str,
    export: &Path,
) -> Result<Command> {
    validate_invocation(invocation)?;
    let mut command = invocation.command();
    command.args([
        "--permission-mode",
        "dangerous",
        "--respect-workspace-trust",
        "false",
        "--export",
    ]);
    command.arg(export);
    if let Some(id) = &invocation.native_session_id {
        command.arg(format!("--resume={id}"));
    } else if !invocation.first_message {
        bail!("Devin resume requires a native session ID");
    }
    if let Some(model) = &invocation.model {
        command.args(["--model", model]);
    }
    command.arg("-p").arg("--").arg(prompt);
    Ok(command)
}

fn read_export(path: &Path) -> Option<(Option<String>, Option<String>)> {
    let value: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let session = cli::extract_session_id(&value).filter(|id| !id.is_empty());
    let answer = value
        .get("steps")
        .and_then(Value::as_array)
        .and_then(|steps| {
            steps.iter().rev().find_map(|step| {
                let step = step.as_object()?;
                if step.get("source").and_then(Value::as_str) != Some("agent") {
                    return None;
                }
                step.get("message")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|message| !message.is_empty())
                    .map(str::to_owned)
            })
        });
    Some((session, answer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AgentKind;

    fn invocation() -> Invocation {
        Invocation {
            agent: AgentKind::Devin,
            executable: Path::new("devin").to_owned(),
            workspace: "/workspace".into(),
            native_session_id: None,
            model: None,
            reasoning_effort: None,
            instructions: None,
            message: "Analyze this".into(),
            first_message: true,
        }
    }

    #[test]
    fn builds_print_command_with_export_and_resume() {
        let export = Path::new("/tmp/export.json");
        let command = build_command(&invocation(), "prompt", export).unwrap();
        let debug = format!("{command:?}");
        for expected in [
            "-p",
            "--permission-mode",
            "dangerous",
            "--respect-workspace-trust",
            "false",
            "--export",
            "/tmp/export.json",
            "--",
        ] {
            assert!(debug.contains(expected), "{expected}: {debug}");
        }
        assert!(!debug.contains("--resume"), "{debug}");

        let mut resume = invocation();
        resume.native_session_id = Some("devin-123".into());
        resume.first_message = false;
        resume.model = Some("opus".into());
        let command = build_command(&resume, "prompt", export).unwrap();
        let debug = format!("{command:?}");
        assert!(debug.contains("--resume=devin-123"), "{debug}");
        assert!(debug.contains("--model"), "{debug}");
        assert!(debug.contains("opus"), "{debug}");

        let mut missing = invocation();
        missing.first_message = false;
        assert!(build_command(&missing, "prompt", export).is_err());

        let mut effort = invocation();
        effort.reasoning_effort = Some("high".into());
        assert!(build_command(&effort, "prompt", export).is_err());
    }

    #[test]
    fn reads_session_and_last_agent_message_from_atif_export() {
        let directory = tempfile::tempdir().unwrap();
        let export = directory.path().join("export.json");
        std::fs::write(
            &export,
            serde_json::json!({
                "schema_version": "ATIF-v1.7",
                "session_id": "devin-session-9",
                "agent": {"name": "devin", "version": "3000.10.31"},
                "steps": [
                    {"step_id": 1, "source": "user", "message": "task"},
                    {"step_id": 2, "source": "agent", "message": "Working on it"},
                    {"step_id": 3, "source": "agent", "tool_calls": []},
                    {"step_id": 4, "source": "agent", "message": "Final answer"},
                    {"step_id": 5, "source": "system", "message": "done"}
                ]
            })
            .to_string(),
        )
        .unwrap();
        let (session, answer) = read_export(&export).unwrap();
        assert_eq!(session.as_deref(), Some("devin-session-9"));
        assert_eq!(answer.as_deref(), Some("Final answer"));

        std::fs::write(&export, "not json").unwrap();
        assert!(read_export(&export).is_none());
        assert!(read_export(&directory.path().join("missing.json")).is_none());
    }
}
