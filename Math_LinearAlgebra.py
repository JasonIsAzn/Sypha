import numpy as np
import sympy as sp

##### Cryptography #####

def letters_to_index(message, alpha_order):
    """
    Turn letters into index of alphabet order

    Parameters:
        message (string/list): message to be indexed
        alpha_order (string): pre-determined alphabet index

    Returns:
        index_list (list): indexed message
    """
    index_list = []
    for i in message:
        for index, j in enumerate(alpha_order):
            if i.lower() == j:
                index_list.append(index)
                break
    return index_list


def index_to_letters(message, alpha_order):
    """
    Turn index into letter according to alphabet order

    Parameter:
        message (string/list): message to be indexed
        alpha_order (string): pre-determined alphabet index

    Returns:
        letter_list (list): letters of indexed message
    """
    letter_list = []
    for i in message:
        for index, j in enumerate(alpha_order):
            if i == index:
                letter_list.append(j)
                break
    return letter_list


def reformatted_txt(txt1, txt2):
    """
    reform text1 to correspond to txt2 format

    Parameter:
        plain_txt (string): plain text
        cipher_txt (string): cipher text

    Returns:
        cipher_txt (string): reformatted cipher text

    """
    for index, character in enumerate(txt1):
        if character == ' ':
            txt2 = txt2[:index] + " " + txt2[index:]

    for index, character in enumerate(txt1):
        if character.isupper():
            txt2 = txt2[:index] + txt2[index].upper() + txt2[index + 1:]

    return txt2


def adjoint_matrix(matrix, num):
    """
    Obtain adjoint of a matrix

    Parameter:
        matrix (array): the matrix to be operated on

    Return:
        adj (array): adjoint of matrix
    """
    if num == 2:
        adj = np.array([[matrix[1][1] % 26, -matrix[0][1] % 26], [-matrix[1][0] % 26, matrix[0][0] % 26]])
        return adj
    elif num > 2:
        cofactors = []
        for i in range(num):
            for j in range(num):
                temp = []
                for row in range(num):
                    for col in range(num):
                        if row != i and col != j:
                            temp.append(matrix[row][col])
                    if len(temp) == num ** 2 - (num * 2) + 1:
                        temp = np.array(temp)
                        temp.resize(num - 1, num - 1)
                        break
                sign = 1 if ((i + j) % 2 == 0) else -1
                cofactors.append(round(sign * np.linalg.det(temp)))
        cofactors = np.array(cofactors)
        cofactors.resize(num, num)
        cofactors = cofactors.T
        return cofactors


def txt_matrix(txt, alpha_order, num):
    """
    Turns plaintext or ciphertext into a matrix

    Parameters:
        txt (string): plain text to turn into list of matrix
        alpha_order (string): alpha_order index
        num (int): matrix size

    Returns:
        plain_matrix (list): list of matrix
    """

    # Adding Filler to PlainText
    txt = txt.replace(" ", '')
    extra_char = int(np.ceil(len(txt) / num) * num) - len(txt)
    txt = txt + (txt[-1] * extra_char)
    num_list = letters_to_index(txt, alpha_order)
    plain_matrix = [np.array([[index] for index in num_list[i:i + num]]) for i in range(0, len(txt), num)]
    return plain_matrix


def defined_key(raw_key, num, alpha_order):
    """
    Turns raw key into useful key

    Parameters:
        raw_key (list): key could be in form of all letters or numbers
        num (int): key size
        alpha_order (string): alpha order index

    Returns:
        refined_key (array): key to be

    """

    invalid_key = True if len(raw_key) != num ** 2 else False
    if invalid_key:
        return "Invalid_key_Len"

    int_exist = False
    letter_exist = False
    for i in raw_key:
        try:
            test = int(i)
            int_exist = True
        except ValueError:
            letter_exist = True
    if int_exist and letter_exist:
        return "Invalid_Key_Comb"

    if int_exist:
        valid_key = [int(i) for i in raw_key]
    else:
        valid_key = letters_to_index(raw_key, alpha_order)

    key_matrix = np.array([(valid_key[i:i + num]) for i in range(0, len(raw_key), num)]).T
    try:
        np.linalg.inv(key_matrix)
        deter = np.linalg.det(key_matrix) % 26
        for x in range(1, 26):
            if (((deter % 26) * (deter % 26)) % 26 == 1):
                inverse_deter = x
        test = inverse_deter
    except:
        return "Invalid_Key_NotInv"

    return key_matrix


def hill_cipher_encoder(plain_matrix, key_matrix, alpha_order):
    """
    Hill Cipher Encoder turns plain text into cipher text using a key

    Parameters:
        plain_matrix(array): plain text in matrix form
        key_matrix(array): key in matrix form

    Returns:
        cipher_text(string): Encoded Message
    """

    cipher_txt = ''

    for group in plain_matrix:
        temp = []
        matrix_mult = key_matrix @ group
        mm_list = matrix_mult.tolist()
        for i in range(len(mm_list)):
            temp.append(mm_list[i][0] % 26)
        cipher_letters = index_to_letters(temp, alpha_order)
        for letter in cipher_letters:
            cipher_txt += letter

    return cipher_txt


def hill_cipher_decoder(cipher_matrix, key_matrix, alpha_order, num):
    """
    Hill Cipher Decomder turn cipher text into plain text using a key

    Parameters:
        cipher_matrix (array): cipher text in matrix form
        key_matrix (array): key in matrix form

     Returns:
         plain_txt(string): Decoded Message
    """

    plain_txt = ''

    deter = np.linalg.det(key_matrix) % 26
    adj = adjoint_matrix(key_matrix, num)

    # Inverse of determinant mod 26
    if num == 2:
        inverse_deter = 1 / deter
    else:
        for x in range(1, 26):
            if (((deter % 26) * (deter % 26)) % 26 == 1):
                inverse_deter = x

    inv_key = inverse_deter * adj
    inv_key_mod = []
    for i in inv_key:
        for j in i:
            inv_key_mod.append(round(j % 26))
    inv_key_mod = np.array(inv_key_mod).reshape(num, num)

    for group in cipher_matrix:
        temp = []
        matrix_mult = inv_key_mod @ group
        mm_list = matrix_mult.tolist()
        for i in range(len(mm_list)):
            temp.append(mm_list[i][0] % 26)
        cipher_letters = index_to_letters(temp, alpha_order)
        for letter in cipher_letters:
            plain_txt += letter

    return plain_txt


def encoder_test(inputs):
    plain_txt, raw_key, num, alpha_order = inputs
    plain_matrix = txt_matrix(plain_txt, alpha_order, num)
    key_matrix = defined_key(raw_key, num, alpha_order)

    if type(key_matrix) == str:
        print(key_matrix)
    else:
        cipher_txt = hill_cipher_encoder(plain_matrix, key_matrix, alpha_order)
        cipher_txt = reformatted_txt(plain_txt, cipher_txt)

        print("Cipher Text =", cipher_txt)
        print("With Key: \n", key_matrix)
        print("\nSuccessful Run")


def decoder_test(inputs):
    cipher_txt, raw_key, num, alpha_order = inputs
    cipher_matrix = txt_matrix(cipher_txt, alpha_order, num)
    key_matrix = defined_key(raw_key, num, alpha_order)

    if type(key_matrix) == str:
        print(key_matrix)
    else:
        plain_txt = hill_cipher_decoder(cipher_matrix, key_matrix, alpha_order, num)
        plain_txt = reformatted_txt(cipher_txt, plain_txt)
        print("Plain Text =", plain_txt)
        print("With Key: \n", key_matrix)
        print("\nSuccessful Run")


alpha0 = 'abcdefghijklmnopqrstuvwxyz'
alpha1 = 'zabcdefghijklmnopqrstuvwxy'
#encoder_test(["Howdy World", ['1', '2', '2', '5'], 2, alpha0])
decoder_test(["Jgchq Cwjrl", ['1', '2', '2', '5'], 2, alpha0])



##### Eigen Values/Vectors #####
def matrix_op(matrix,  line, A, B, factor_A, factor_B=1):
    """
    Matrix operations that involves adding/subtracting from/to a given line by a given factor.

    Parameters:
        matrix (np.array): The original Matrix
        line (string): "column"/"row"
        A (int): The line to be factored and operated into line_B
        B (int): The line to be replaced
        factor_A (int/eqn): The factor for line_A
        factor_B (int/eqn): The factor for line_B. Changed if line_A = '0'

    Returns:
        op_matrix (array): The operated Matrix
    """

    if line == "column":
        matrix = matrix.T

    line_A = matrix[A - 1] * factor_A
    line_B = matrix[B - 1] * factor_B

    new_line = line_A + line_B

    op_matrix = []
    for index, x in enumerate(matrix):
        if index + 1 == int(B):
            op_matrix.append(new_line)
        else:
            op_matrix.append(x)

    op_matrix = np.array(op_matrix)

    if line == "column":
        op_matrix = op_matrix.T

    return op_matrix


def matrix_op_swap(matrix, line, A, B):
    """
    Matrix operation to swap line

    Parameters:
        matrix (np.array): The given array
        line (string): column/row
        A (int): line A to swap with line B
        B (int): line B to swap with line A

    Returns:
        swap_matrix()
    """
    swap_matrix = []

    if line == "column":
        matrix = matrix.T

    for index, item in enumerate(matrix):
        if index + 1 == A:
            swap_matrix.append(matrix[B - 1])
        elif index + 1 == B:
            swap_matrix.append(matrix[A - 1])
        else:
            swap_matrix.append(item)

    swap_matrix = np.array(swap_matrix)

    if line == "column":
        swap_matrix = swap_matrix.T

    return swap_matrix


def matrix_op_var(matrix, line, A, B, factor_A, factor_B=1):
    """
    matrix_op but working with symbols

    Parameters:
        matrix (np.array): The original Matrix
        op (int): -1 or 1
        line (string): "column"/"row"
        A (int): The line to be factored and operated into line_B
        B (int): The line to be replaced
        factor_A (int/eqn): The factor for line_A
        factor_B (int/eqn): The factor for line_B. Changed if line_A = '0'

    Returns:
        op_matrix (array): The operated Matrix
    """

    if line == "column":
        matrix = matrix.T

    line_A = []
    line_B = []
    for item in matrix[A - 1]:
        line_A.append(item * factor_A)
    for item in matrix[B - 1]:
        line_B.append(item * factor_B)

    new_line = []
    for num_A, num_B in zip(line_A, line_B):
        new_line.append(-1 * num_A + num_B)
    op_matrix = []
    for index, x in enumerate(matrix):
        if index + 1 == int(B):
            op_matrix.append(new_line)
        else:
            op_matrix.append(x)

    op_matrix = np.array(op_matrix)

    if line == "column":
        op_matrix = op_matrix.T

    return op_matrix


def matrix_lambda(matrix):
    """
    subtract lambda from diagonal in square matrix

    Parameters:
        matrix (np.array): The original matrix

    Returns:
        lambda_matrix (array): Matrix with lambda

    """
    y = sp.symbols('y')

    lambda_matrix = []
    for i, row in enumerate(matrix):
        new_row = []
        for j, num in enumerate(row):
            if i == j:
                new_row.append(num - y)
            else:
                new_row.append(num)
        lambda_matrix.append(new_row)
    lambda_matrix = np.array(lambda_matrix)

    return lambda_matrix


def deter2x2(matrix):
    """
    Given a 2x2 matrix, return determinant using simple formula
    Parameters:
        matrix (np.array): The give matrix
    Results:
        deter (sq.eqn/int): A value or an equation
    """
    deter = (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
    return deter


def matrix_determinant(matrix):
    """
    Given a matrix, return the determinant of the matrix

    Parameters:
        matrix (np.array): The given matrix

    Results:
        deter (sp.eqn): An equation

    """
    l_matrix = matrix_lambda(matrix)
    deter = 1
    if len(l_matrix) == 2:
        deter = deter2x2(l_matrix)
        return deter

    temp_matrix = l_matrix
    for num in range(len(temp_matrix[0]) - 2):
        deter *= temp_matrix[0][0]
        for index, item in enumerate(temp_matrix[0]):
            if index == 0:
                pass
            else:
                temp_matrix = matrix_op_var(temp_matrix, "column", 1, index + 1, item/temp_matrix[0][0])

        expand_matrix = []
        for index, item in enumerate(temp_matrix):
            if index == 0:
                pass
            else:
                item = item.tolist()
                items = item[1:len(item) + 1]
                expand_matrix.append(items)
        expand_matrix = np.array(expand_matrix)
        temp_matrix = expand_matrix
    deter *= deter2x2(temp_matrix)
    deter = sp.simplify(deter).factor()
    return deter


def rref(matrix):
    """
    reduced row echelon form. Any Size Matrix

    Parameters:
        matrix (np.array): Given Matrix

    Returns:
        rref_matrix (np.array): rref matrix, identify matrix on the left

    """
    rref_matrix = matrix.T

    stop = False
    for i in range(len(matrix.T)):
        column = rref_matrix[i]
        if stop:
            pass
        else:
            for j in range(len(matrix.T[0])):
                num = column[j]
                if j == 0:
                    if column[i] == 0 and i != len(matrix.T[0]) - 1:
                        rref_matrix = matrix_op_swap(rref_matrix, "column", i + 1, i + 2)
                        column = rref_matrix[i]
                        num = column[j]
                    rref_matrix = matrix_op(rref_matrix, "column", i + 1, i + 1, 0, 1/column[i])
                if i == j or num == 0:
                    continue
                else:
                    rref_matrix = matrix_op(rref_matrix, "column", i + 1, j + 1, -num)
                for line in rref_matrix:
                    if line[-1] != 0:
                        stop = False
                        pass
                    else:
                        stop = True

    rref_matrix = rref_matrix.T

    return rref_matrix


def matrix_eigen_vectors(matrix, eigen_values):
    """
    Find EigenVector given a square matrix

    Parameters:
        matrix (np.array): Given Matrix

    Returns:
         e_vector (np.array): eigen vector
    """

    l_matrix = matrix_lambda(matrix)
    vectors = []
    for eigen in eigen_values:
        ll_matrix = []
        for i, line in enumerate(l_matrix):
            temp = []
            for j, num in enumerate(line):
                if i == j:
                    y = sp.symbols('y')
                    temp.append(num.subs(y, eigen))
                else:
                    temp.append(num)
            ll_matrix.append(temp)
        ll_matrix = np.array(ll_matrix)
        rref_ll = rref(ll_matrix).T
        last_row = rref_ll[-1]
        vector = []
        for i, num in enumerate(last_row):
            if i == len(last_row) - 1:
                vector.append(1)
            else:
                vector.append(-1 * num)
        vectors.append(vector)

    vectors = np.array(vectors)

    return vectors


def matrix_eigen(matrix):
    """
    Finds EigenValue and EigenVector given a square matrix

    Parameters:
        matrix (np.array): Given Matrix
    Returns:
        ans (list): Contain list of EigenValue and EigenVector in Matrix form
    """

    deter = matrix_determinant(matrix)
    eigen_values = sp.solve(deter)
    eigen_vectors = matrix_eigen_vectors(matrix, eigen_values)
    ans = eigen_values, eigen_vectors
    return ans


def matrix_eigen_test():
    matrix = np.array([[3, 5, -5, 5],
                       [3, 1, 3, -3],
                       [-2, 2, 0, 2],
                       [0, 4, -6, 8]])

    eigen_values, eigen_vectors = matrix_eigen(matrix)
    print("Given the following matrix: ")
    print(matrix)
    print("\nThe eigen values are", eigen_values)
    print("\nWith the corresponding eigen Vectors:")
    print(eigen_vectors)

matrix_eigen_test()



##### QR and LU Decomposition #####
