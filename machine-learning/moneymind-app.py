import streamlit as st

def main():
    # Streamlit app layout
    st.title("💸 Financial Literacy Chatbot")
    st.write("""
        Welcome to the Financial Literacy Chatbot! Ask questions about budgeting, saving, and personal finance tips.
        The advice here is inspired by **The Richest Man in Babylon**, a classic book on personal finance principles.
    """)
    
    # Chatbot interaction section
    st.header("Chat with the Financial Literacy Bot 🤖")
    user_input = st.text_input("What would you like to know?", "")
    
    if user_input:
        # Generate response
        with st.spinner("Thinking..."):
            response = chatbot_response(user_input)
        
        # Display response
        st.success("Here's some advice:")
        st.write(response)
    
    # Additional tips section
    st.sidebar.title("💡 Financial Tips")
    st.sidebar.write("🔸 Save a fixed percentage of your income each month.")
    st.sidebar.write("🔸 Avoid unnecessary debt.")
    st.sidebar.write("🔸 Invest wisely to grow your wealth.")
    st.sidebar.write("🔸 Track expenses and create a budget.")

if __name__ == "__main__":
    main()
