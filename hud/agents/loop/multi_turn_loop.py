"""Multi-turn agent interaction framework for HUD."""

import asyncio
import logging
from typing import Any

from hud.agents.base import text_to_blocks
from hud.eval.context import EvalContext
from hud.types import Trace

logger = logging.getLogger(__name__)


def _check_stop_signal(text: str, stop_signal: str = "###STOP###") -> bool:
    """Check if text contains the conversation stop signal."""
    if stop_signal in text:
        logger.info(f"Detected stop signal: {stop_signal}")
        return True
    return False


async def multi_turn_run(
    ctx: EvalContext,
    agent: Any,
    simulated_user: Any,
    max_steps: int = 30,
) -> Trace:
    """
    Run multi-turn conversation between agent and simulated user.

    Drop-in replacement for `await agent.run(ctx)`.
    Conversation ends when user sends ###STOP### signal.
    """
    if not isinstance(ctx, EvalContext):
        raise TypeError(f"ctx must be EvalContext, got {type(ctx).__name__}")
    if not ctx.prompt:
        raise ValueError("ctx.prompt is not set")

    # Setup agents
    agent.ctx = simulated_user.ctx = ctx
    await _initialize_agent_with_filters(agent, ctx)
    await _initialize_agent_with_filters(simulated_user, ctx)

    try:
        result = await _run_conversation_loop(
            agent, simulated_user, text_to_blocks(ctx.prompt), max_steps=max_steps
        )
        if result.content and ctx.has_scenario:
            await ctx.submit(result.content)
        return result
    except Exception as e:
        logger.exception("Multi-turn agent error:")
        return Trace(done=True, content=f"Error: {e}", isError=True, info={"error": str(e)})
    finally:
        await agent._cleanup()
        await simulated_user._cleanup()


async def _initialize_agent_with_filters(agent: Any, ctx: EvalContext) -> None:
    """Initialize agent and apply tool filtering if configured."""
    if agent._initialized:
        return

    await agent._initialize_from_ctx(ctx)

    # Apply allowed_tools filter if configured
    if hasattr(agent.config, 'allowed_tools') and agent.config.allowed_tools:
        agent._available_tools = [
            t for t in agent._available_tools if t.name in agent.config.allowed_tools
        ]
        agent._tool_map = {t.name: t for t in agent._available_tools}
        agent.console.info(
            f"Filtered to {len(agent._available_tools)} tools: "
            f"{', '.join(t.name for t in agent._available_tools)}"
        )
        agent._on_tools_ready()


async def _run_conversation_loop(
    agent: Any,
    simulated_user: Any,
    context: list[Any],
    *,
    max_steps: int = 30,
) -> Trace:
    """Core conversation loop with turn-based interaction."""
    final_response = None
    error = None
    messages: list[Any] = []

    async def get_user_response(agent_message: str) -> str:
        """Get simulated user response to agent's message."""
        try:
            # Run user agent with agent's message as prompt
            user_prompt = f"The assistant said: {agent_message}\n\nRespond as a user."
            user_messages = await simulated_user.get_system_messages()
            user_messages.extend(await simulated_user.format_message(user_prompt))

            # User can call tools and respond
            max_user_iterations = 6
            for _ in range(max_user_iterations):
                user_response_obj = await simulated_user.get_response(user_messages)

                # If user has tool calls, execute them
                if user_response_obj.tool_calls:
                    logger.info(f"User executing {len(user_response_obj.tool_calls)} tool(s)")
                    user_tool_results = await simulated_user.call_tools(user_response_obj.tool_calls)

                    # Format tool results and add to messages
                    tool_messages = await simulated_user.format_tool_results(
                        user_response_obj.tool_calls, user_tool_results
                    )
                    user_messages.extend(tool_messages)

                    # Continue to get text response after tools
                    continue
                else:
                    # No tool calls - user has a text response
                    return user_response_obj.content or "Okay."

            # Max iterations reached - return last content
            return user_response_obj.content or "Okay."

        except TimeoutError:
            logger.error("User response timed out")
            return "Sorry, I took too long to respond."
        except Exception as e:
            logger.error(f"Failed to get user response: {e}")
            import traceback
            traceback.print_exc()
            return f"Error getting user response: {e}"

    try:
        # Start with system messages
        messages = await agent.get_system_messages()

        # Add initial context
        messages.extend(await agent.format_message(context))
        agent.console.debug(f"Messages: {messages}")

        step_count = 0
        while max_steps == -1 or step_count < max_steps:
            step_count += 1
            agent.console.debug(f"Step {step_count}/{max_steps if max_steps != -1 else 'unlimited'}")

            try:
                # 1. Get agent response
                response = await agent.get_response(messages)
                agent.console.debug(f"Agent:\n{response}")

                # 2. Check if agent has tool calls
                if response.tool_calls:
                    # Execute agent tools
                    tool_calls = response.tool_calls
                    tool_results = await agent.call_tools(tool_calls)

                    # Format tool results and add to messages
                    tool_messages = await agent.format_tool_results(tool_calls, tool_results)
                    messages.extend(tool_messages)

                    # Display
                    step_info = f"\n[bold]Step {step_count}/{max_steps if max_steps != -1 else '∞'}[/bold]"
                    for call, result in zip(tool_calls, tool_results, strict=False):
                        step_info += f"\n🤖 {call}\n{result}"
                    agent.console.info_log(step_info)

                    # Check if agent also sent a message (conversation turn)
                    if response.content:
                        agent_message = response.content
                        agent.console.info(f"[bold cyan]🤖 Agent:[/bold cyan] {agent_message}")

                        # Get user response
                        user_response = await get_user_response(agent_message)
                        agent.console.info(f"[bold green]👤 User:[/bold green] {user_response}")

                        # Check for stop signal in user response
                        if _check_stop_signal(user_response):
                            agent.console.info("Conversation ended by user signal")
                            final_response = response
                            break

                        # Add user response to messages
                        messages.extend(await agent.format_message(user_response))

                else:
                    # No tool calls - agent sent message to user
                    agent_message = response.content or ""

                    if not agent_message:
                        # Agent provided empty response
                        agent.console.warning("Agent provided empty response, ending")
                        final_response = response
                        break

                    agent.console.info(f"[bold cyan]🤖 Agent:[/bold cyan] {agent_message}")

                    # Add agent message to history (format as string, not AgentResponse)
                    messages.extend(await agent.format_message(agent_message))

                    # Get user response
                    user_response = await get_user_response(agent_message)
                    agent.console.info(f"[bold green]👤 User:[/bold green] {user_response}")

                    # Check for stop signal in user response
                    if _check_stop_signal(user_response):
                        agent.console.info("Conversation ended by user signal")
                        final_response = response
                        break

                    # Add user response to messages and continue
                    messages.extend(await agent.format_message(user_response))

            except Exception as e:
                agent.console.error_log(f"Step failed: {e}")
                error = str(e)
                break

    except KeyboardInterrupt:
        agent.console.warning_log("Agent execution interrupted by user")
        error = "Interrupted by user"
    except asyncio.CancelledError:
        agent.console.warning_log("Agent execution cancelled")
        error = "Cancelled"
    except Exception as e:
        agent.console.error_log(f"Unexpected error: {e}")
        error = str(e)

    # Build result
    is_error = error is not None or (
        final_response and hasattr(final_response, "isError") and final_response.isError
    )

    trace_params = {
        "reward": 0.0,
        "done": True,
        "messages": messages,
        "content": final_response.content if final_response else (error or "Conversation ended"),
        "isError": is_error,
        "info": {"error": error} if error else {},
    }
    trace_result = Trace(**trace_params)

    return trace_result
