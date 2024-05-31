"""
    _summary_
    
    Module use to operate the Authentication backend, Authentication API
    include: Login, Register users, Account Page. Saved project
    
    Read Code 2.1, Code 2.2, Code 2.3, Code 2.4 for more information.


"""


from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User


def login_render(request):
    """"_summary_
    Code 2.1


    Args:
        request (GET): Recieve a http/https request and return a page

    Returns:
        _type_: _description_
    """
    return render(request, 'login.html')


def register_render(request):
    """_summary_
    Code 2.2

    Args:
        request (GET): Recieve a http/https request and return a page

    Returns:
        HttpResponse: Returning a Registeration page
    """
    return render(request, 'register.html')

@login_required(login_url='login')
def account_render(request):
    """_summary_
    Code 2.3

    Args:
        request (GET): Recieve a http/https request and return a page

    Returns:
        HttpResponse: Returning a Account page
    """
    return render(request, 'account.html')

@login_required(login_url='login')
def savedproject_render(request):
    """_summary_
    Code 2.4

    Args:
        request (GET): Recieve a http/https request and return a page

    Returns:
        HttpResponse: Returning a Saved Project page
    """
    return render(request, 'savedprojects.html')


def login_agent(request):
    """
    Logs in an agent.

    Args:
        request (HttpRequest): The HTTP request containing login credentials.

    Returns:
        HttpResponseRedirect: Redirects to the dashboard upon successful login.
        HttpResponseRedirect: Redirects to the login page if login fails or if credentials are missing.
    """

    if request.method == "POST":
        # Extract email and password from the request
        email = request.POST.get("email")
        password = request.POST.get("password")
        # Check if both email and password are provided
        if email and password:
            # Authenticate the user
            user = authenticate(username=email, password=password)
            # Check if authentication is successful
            if user is not None:
                # Login the user
                login(request, user)

                return redirect('dashboard')
        else:
            # Redirect to the login page if either email or password is missing
            return redirect("login")

    # Redirect to the login page to try again of not a POST request
    return redirect("login")


def register_agent(request):
    """
    Registers a new agent.
    
    Args:
        request (HttpRequest): The HTTP request containing user information.

    Returns:
        HttpResponseRedirect: Redirects to the login page upon successful registration.
        HttpResponseRedirect: Redirects to the login page if registration fails.
    """

    if request.method == "POST":
        # Extract user information from the request
        firstname = request.POST.get("firstname")
        lastname = request.POST.get("lastname")
        email = request.POST.get("email")
        password = request.POST.get("pass1")
        confirm_password = request.POST.get("pass2")

        # Check if all required fields are provided
        if firstname and lastname and email and password and confirm_password:
            # Check if passwords match
            if password == confirm_password:
                # Create a new user object
                user = User.objects.create(first_name=firstname, last_name=lastname, username=email)
                # Set the password for the user
                user.set_password(password)
                # Save the user object
                user.save()
                # Redirect to the login page upon successful registration
                return redirect('login')
        else:
            # Redirect to the login page if any required fields are missing
            return redirect('register')

    # Redirect to the login page if the request method is not POST
    return redirect('register')

def logout_agent(request):
    """_summary_
    Code 2.7

    Args:
        request (GET): Recieve a http/https request and Remove users sessional cookies

    Returns:
        HttpRedirect: After logout return to login page
    """
    logout(request)
    return redirect('login')