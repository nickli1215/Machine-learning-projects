import java.awt.*;
import java.util.*;

public class Piece 
{
	Color purple = new Color (255, 0, 255);
	Color[] colors = new Color[] {null, Color.orange, Color.blue, Color.red, Color.green,
	purple, Color.YELLOW, Color.cyan};
	private int cy;
	private int cx;
	// I is cyan, square is yellow, T is purple, S is red, Z is green, J is orange, and L is dark blue
	private int shapenum;
	private int rotation;
	// 0 is I, 1 is square, 2 is T, 3 is S, 4 is Z, 5 is J, 6 is L
	// Each shape is represented by a 2D array of Points, which each contain rotation arrays
	private final Point[][][] shapes = 
		{
				// J
			{	
				{ new Point(0, 0), new Point(0, 1), new Point(0, -1), new Point(1,-1) },
				{ new Point(0, 0), new Point(-1, 0), new Point(-1, -1), new Point(1, 0) },
				{ new Point(0, 0), new Point(0, 1), new Point(-1, 1), new Point(0, -1) },
				{ new Point(-1, 0), new Point(0,0), new Point(1, 0), new Point(1, 1) }
			},

			//L
			{
				{ new Point(0,0), new Point (0,1), new Point (-1,-1), new Point (0,-1)},
				{ new Point(0,0), new Point (1,0), new Point(-1,0), new Point (1,-1)},
				{ new Point (0,0), new Point (0,1), new Point (1,1), new Point (0,-1)},
				{ new Point(0,0), new Point (-1,0), new Point(-1,1), new Point (1,0)}
			}, 
				//Z
			{
				{new Point (-1,0), new Point (0,0), new Point (0,1), new Point (1,1)},
				{new Point (1,0), new Point (0,0), new Point (0,1), new Point (1,-1)},
				{new Point (-1,0), new Point (0,0), new Point (0,1), new Point (1,1)},
				{new Point (1,0), new Point (0,0), new Point (0,1), new Point (1,-1)}
			},
				//S
			{
				{new Point (-1,1), new Point (0,1), new Point (0,0), new Point (1,0)},
				{new Point (-1,-1), new Point (-1,0), new Point (0,1), new Point (0,0)},
				{new Point (-1,1), new Point (0,1), new Point (0,0), new Point (1,0)},
				{new Point (-1,-1), new Point (-1,0), new Point (0,1), new Point (0,0)}
			},
				//T
			{
				{new Point (0,0), new Point (0,1), new Point (1,0), new Point (-1,0)},
				{new Point (0,0), new Point (0,1), new Point (0,-1), new  Point (1,0)},
				{new Point (0,0), new Point (1,0), new Point (-1,0), new Point (0,-1)},
				{new Point (0,0), new Point (0,1), new Point (0,-1), new Point (-1,0)}
			},
				// O
			{
				{new Point (0,0), new Point (1,0), new Point (1,1), new Point (0,1)},
				{new Point (0,0), new Point (1,0), new Point (1,1), new Point (0,1)},
				{new Point (0,0), new Point (1,0), new Point (1,1), new Point (0,1)},
				{new Point (0,0), new Point (1,0), new Point (1,1), new Point (0,1)}
			},
				//I
			{
				{new Point (-1,0), new Point(0,0), new Point (1,0), new Point (2,0)},
				{new Point (0,-1), new Point (0,0), new Point (0,1), new Point (0,2)},
				{new Point (-1,0), new Point(0,0), new Point (1,0), new Point (2,0)},
				{new Point (0,-1), new Point (0,0), new Point (0,1), new Point (0,2)}
			}
		};
	//shape stores a set of points used to display the shape from the centre
	private Point[] shape;
	public Piece (Point centre, int shapenum, int r)
	{
		cx = (int)centre.getX();
		cy = (int)centre.getY();
		shape = shapes[shapenum][r];
		this.rotation = r;
		this.shapenum =shapenum+1;
	}
	//copies the piece's color value to the board
	public void copy (int[][] board)
	{
		for (Point i : shape)
			board[(int)(i.getY()+cy)][(int)(i.getX() + cx)] = shapenum;
	}
	
	public int getShape()
	{
		return shapenum-1;
	}
	//rotates m by choosing a new rotation value within the 
	public void rotate(int[][] board,int x)
	{
		int m = (rotation + x)%4;
		if(m<0)
			m = 3;
		//checking if rotation is possible
		if(check(board,shapes[shapenum-1][m],cy,cx))
		{
			shape = shapes[shapenum-1][m];
			rotation = m;
		}
		
	}

	public void display ( Graphics g, int size)
	{
		//Draws in the points in reference to the centre
		for (Point i : shape)
		{
			if((int)(i.getY()+cy)>3)
			{
				g.setColor(colors[shapenum]);
				g.fillRect(size*(int)(i.getX() + cx),size*(int)(i.getY()+cy+3),size,size);
				g.setColor(Color.black);
				g.drawRect(size*(int)(i.getX() + cx),size*(int)(i.getY()+cy+3),size,size);
			}
		}
	}
	public int isEmpty(int[][] arr, int i)
	{
		int max = -1; 
		for (int x = 0; x<arr[0].length; x++)
		{
			if(arr[x][i] >0)
				max = x;
		}
		return max;
	}
	//returns false if its impossible to move there, returns true otherwise
	public boolean check (int[][] board,Point[] shape, int cy, int cx)
	{
		for (Point j : shape)
		{
			int x = cx +(int)j.getX();
			int y = cy +(int)j.getY();
			if(y>=board.length||x>=board[0].length||x<0)
				return false;
			try
			{
				if(board[y][x] >0)
				{
					return false;
				}
			}
			catch (IndexOutOfBoundsException e)
			{
				return false;
			}
		}
		return true;
	}
	//returns true if the piece can fall, based on check function
	public boolean fall (int[][] board)
	{
		if(check(board,shape,cy+1,cx))
		{
			cy++;
			return true;
		}
		return false;
	}
	//drops the piece to lowest possible y-value
	public void drop (int[][] board)
	{
		while(check(board,shape,cy+1,cx))
		{
			cy++;
		}
		copy(board);
	}
	// checks if it is possible to move in a certain direction
	public void move(int[][] board, int x, int y)
	{
		if(check(board, shape,cy+y, cx+x))
		{
			cy+=y;
			cx+=x;
		}
	}
}