import asyncio
import logging
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from pydantic import BaseModel, Field
from typing import Any

class CalculatorInput(BaseModel):
    """Input model for basic mathematical calculations."""
    expression: str = Field(description = "Mathematical expression using +, -, *, / (e.g., '10 + 5', '20 - 8', '4 * 6', '15 / 3')")
    
class SimpleCalculatorTool(Tool[CalculatorInput, ToolRunOptions, StringToolOutput]):
    """A simple calculator tool for basic arithmetic operations: add, subtract, multiply, divide."""
    
    name = "SimpleCalculator"
    description = "Performs basic arithmetic calculations: addition (+), subtraction (-), multiplication (*), and division (/)."
    input_schema = CalculatorInput
    
    def init(self, options: dict[str, Any] | None = None) -> None:
        super().init(options)
        
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace = ["tool", "calculator", "basic"], creator = self)
        
    def _safe_calculate(self, expression: str) -> float:
        """Safely evaluate basic arithmetic expressions."""
        expr = expression.replace(' ', '')
        allowed_chars = set('0123456789+-*/().')
        if not all(c in allowed_chars for c in expr):
            raise ValueError("Only numbers and basic operators (+, -, *, /, parentheses) are allowed")
            
        try:
            result = eval(expr, {"builtins": {}}, {})
            return float(result)
        except ZeroDivisionError:
            raise ValueError("Division by zero is not allowed")
        except Exception as e:
            raise ValueError(f"Invalid arithmetic expression: {str(e)}")
            
    async def _run(self, input: CalculatorInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        """Perform basic arithmetic calculations."""
        try:
            expression = input.expression.strip()
            result = self._safe_calculate(expression)
            output = f"🧮 Simple Calculator\n"
            output += f"Expression: {expression}\n"
            output += f"Result: {result}\n"
            
            if '+' in expression:
                output += "Operation: Addition"
            elif '-' in expression:
                output += "Operation: Subtraction"
            elif '*' in expression:
                output += "Operation: Multiplication"
            elif '/' in expression:
                output += "Operation: Division"
            else:
                output += "Operation: Basic Arithmetic"
            return StringToolOutput(output)
            
        except ValueError as e:
            return StringToolOutput(f"❌ Calculation Error: {str(e)}")
        except Exception as e:
            return StringToolOutput(f"❌ Unexpected Error: {str(e)}")
            
async def calculator_agent_example():
    """RequirementAgent with SimpleCalculatorTool - Interactive Math Assistant"""
    
    llm = ChatModel.from_name("watsonx:meta-llama/llama-4-maverick-17b-128e-instruct-fp8", ChatModelParameters(temperature = 0))

    calculator_agent = RequirementAgent(llm = llm,
                                        tools = [SimpleCalculatorTool()],
                                        memory = UnconstrainedMemory(),
                                        instructions = """You are a helpful math assistant. When users ask for calculations, 
                                                          use the SimpleCalculator tool to provide accurate results. 
                                                          Always show both the expression and the calculated result.""",
                                        middlewares = [GlobalTrajectoryMiddleware(included = [Tool])])

    math_queries = ["What is 15 + 27?", "Calculate 144 divided by 12", "I need to know what 8 times 9 equals", "What's (10 + 5) * 3 - 7?"]
    
    for query in math_queries:
        print(f"\n👤 Human: {query}")
        result = await calculator_agent.run(query)
        print(f"🤖 Agent: {result.answer.text}")
        
async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    await calculator_agent_example()

if __name__ == "__main__":
    asyncio.run(main())
